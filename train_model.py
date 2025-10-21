"""
Toy Vision Language Model (VLM) in PyTorch
A simple VLM that can understand basic shapes and answer questions about them.
"""
import os
import math
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from shapes import ShapeGenerator
from questions import QuestionGenerator
from text import TextProcessor
from model import ToyVLM
from device import DEVICE, select_amp, get_autocast_cm

 

class ShapeDataset(Dataset):
    """Dataset that generates simple geometric shapes with Q&A pairs."""
    
    def __init__(self, num_samples: int, text_processor: TextProcessor):
        self.num_samples = num_samples
        self.shape_generator = ShapeGenerator()
        self.question_generator = QuestionGenerator()
        self.text_processor = text_processor
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random shape
        shape_type, image = self.shape_generator.generate_random_shape()
        
        # Generate Q&A pair
        question, answer = self.question_generator.generate_qa_pair(shape_type)
        
        # Prepare sequences using text processor
        input_tokens, target_tokens, loss_mask = self.text_processor.prepare_input_sequence(question, answer)
        
        return {
            'image': torch.tensor(image, dtype=torch.float32).unsqueeze(0),  # Add channel dim
            'input_tokens': torch.tensor(input_tokens, dtype=torch.long),
            'target_tokens': torch.tensor(target_tokens, dtype=torch.long),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.bool),
            'question': question,
            'answer': answer
        }


def init_distributed(backend: str = "nccl"):
    """
    Single-node DDP init. Assumes launch via torchrun --standalone --nproc_per_node=K.
    Relies on env:// and LOCAL_RANK; no multi-node rendezvous or NODE_RANK/WORLD_SIZE management.
    """
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def is_distributed():
    return dist.is_available() and dist.is_initialized()

def train_model(model, train_loader, num_epochs, warmup_steps, total_steps, learning_rate, sampler=None):
    """Train the VLM model."""
    rank = dist.get_rank() if is_distributed() else 0
    is_main = (rank == 0)

    # Ensure CUDA current device is set per process when available
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    model = model.to(DEVICE)
    # Access underlying module attributes when wrapped in DDP
    base_model = model.module if isinstance(model, DDP) else model
    tokenizer = base_model.text_processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    vocab_size = tokenizer.get_vocab_size()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    amp_conf = select_amp(DEVICE)
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_conf['device_type'] == 'cuda' and amp_conf['dtype'] == torch.float16))

    if is_main:
        print(
            f"Autocast selected: device_type={amp_conf['device_type']}, "
            f"dtype={amp_conf['dtype']}, grad_scaler={'on' if scaler.is_enabled() else 'off'}"
        )

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        # cosine decay after warmup
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    if is_main:
        print(f"Training on {DEVICE}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        if sampler is not None:
            sampler.set_epoch(epoch)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=not is_main)
        
        for batch in progress_bar:
            images = batch['image'].to(DEVICE, non_blocking=True)
            input_tokens = batch['input_tokens'].to(DEVICE, non_blocking=True)
            target_tokens = batch['target_tokens'].to(DEVICE, non_blocking=True)
            masked_targets = target_tokens.masked_fill(
                ~batch['loss_mask'].to(DEVICE, non_blocking=True), pad_token_id
            )

            # Forward + loss under autocast if available
            autocast_cm = get_autocast_cm(amp_conf)
            with autocast_cm:
                logits = model(images, input_tokens)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    masked_targets.reshape(-1),
                    ignore_index=pad_token_id
                )
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if is_main:
                progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
        
        if is_main:
            avg_loss = total_loss / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}")
        
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true", help="Enable DDP via torchrun (single-node)")
    parser.add_argument("--backend", type=str, default="nccl", help="NCCL for CUDA, gloo for CPU-only")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-process batch size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=None, help="Total synthetic samples (defaults to 500 * batch size)")
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="toy_vlm.pth")
    args = parser.parse_args()

    # Derive dependent values
    if args.samples is None:
        args.samples = 1000 * args.batch_size
    total_steps = args.epochs * args.samples // args.batch_size

    # Optional perf knobs
    torch.backends.cudnn.benchmark = True

    rank, world_size = 0, 1
    if args.distributed:
        rank, world_size = init_distributed(args.backend)

    is_main = (rank == 0)
    if is_main:
        print("Initializing Toy VLM...")

    # Build tokenizer vocabulary from questions (same on all ranks)
    if is_main:
        print("Building tokenizer vocabulary...")
    question_gen = QuestionGenerator()
    text_processor = TextProcessor()
    text_processor.tokenizer.build_vocab_from_questions(question_gen)
    
    # Save the tokenizer vocabulary only on rank 0
    if is_main:
        text_processor.tokenizer.save_vocab('tokenizer_vocab.json')
    
    # Create model with the built tokenizer
    model = ToyVLM(text_processor)

    # Wrap in DDP if distributed
    if args.distributed:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            model = DDP(model.to(DEVICE), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            model = DDP(model.to(DEVICE))  # CPU/Gloo fallback
    
    # Create dataset and loader
    train_dataset = ShapeDataset(num_samples=args.samples, text_processor=text_processor)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if args.distributed else None
    shuffle = False if sampler is not None else True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0)
    )
    
    # Train model
    model = train_model(
        model,
        train_loader,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        learning_rate=args.learning_rate,
        sampler=sampler,
    )
    
    # Save model from rank 0 only
    if is_main:
        module = model.module if isinstance(model, DDP) else model
        torch.save(module.state_dict(), args.save_path)
        print("Training complete. Model and tokenizer saved.")

    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()