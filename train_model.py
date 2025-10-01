"""
Toy Vision Language Model (VLM) in PyTorch
A simple VLM that can understand basic shapes and answer questions about them.
Now with chain-of-thought reasoning capabilities.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from typing import Dict, List, Any
import argparse
from shapes import ShapeGenerator, ObjType
from questions import RationaleGenerator
from text import TextProcessor, MAX_SEQ_LEN
from model import ToyVLM, DEVICE
import random

# Training constants
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 2e-4

class ShapeDataset(Dataset):
    """Dataset that generates multi-shape images with Q&A pairs and rationales."""

    def __init__(self, num_samples: int, text_processor: TextProcessor, difficulty: str):
        self.num_samples = num_samples
        self.shape_generator = ShapeGenerator()
        self.rationale_generator = RationaleGenerator()
        self.text_processor = text_processor
        self.difficulty = difficulty

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate multi-shape RGB image with metadata
        num_shapes = random.randint(1, 4)
        image, metadata_list = self.shape_generator.generate_multi_shape_image(num_shapes, True)

        # Generate Q&A pair with rationale based on metadata
        question, answer, rationale = self.rationale_generator.generate_qa_with_rationale(
            metadata_list, difficulty=self.difficulty
        )

        inp_ids, tgt_ids, rat_mask, ans_mask = self.text_processor.prepare_input_sequence(
            question, answer, rationale
        )

        # Generate auxiliary labels from metadata
        aux_labels = self._generate_aux_labels(metadata_list)

        # ---- truncate to MAX_SEQ_LEN together (before padding) ----
        def trunc(x, L=MAX_SEQ_LEN): return x[:L]
        inp_ids   = trunc(inp_ids);   tgt_ids   = trunc(tgt_ids)
        rat_mask  = trunc(rat_mask); ans_mask = trunc(ans_mask)

        # ---- pad everything ----
        inp_ids   = self.text_processor.pad_sequence(inp_ids)
        tgt_ids   = self.text_processor.pad_sequence(tgt_ids)
        rat_mask  = self.text_processor.pad_sequence(rat_mask)
        ans_mask  = self.text_processor.pad_sequence(ans_mask)

        # ---- image normalization (grayscale) ----
        # Grayscale image: (H,W) -> (1,H,W)
        img = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0 # (1,H,W)

        # Shape and normalization assertions
        assert img.dtype == torch.float32, f"Image dtype should be float32, got {img.dtype}"
        assert img.shape == (1, 64, 64), f"Image shape should be (1, 64, 64), got {img.shape}"
        assert 0.0 <= img.min() and img.max() <= 1.0, \
            f"Image should be normalized to [0,1], got range [{img.min():.3f}, {img.max():.3f}]"

        return {
            'image': img,
            'input_tokens': torch.tensor(inp_ids, dtype=torch.long),
            'target_tokens': torch.tensor(tgt_ids, dtype=torch.long),
            'rat_mask': torch.tensor(rat_mask, dtype=torch.float32),
            'ans_mask': torch.tensor(ans_mask, dtype=torch.float32),
            'question': question, 'answer': answer, 'rationale': rationale,
            'aux_labels': aux_labels, 'metadata': metadata_list
        }


    def _generate_aux_labels(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Generate ground truth labels for auxiliary heads."""
        # Count each shape type (for count heads)
        shape_counts = {obj_type.value: 0 for obj_type in ObjType}

        # Count each size category
        size_counts = {size: 0 for size in ['small', 'medium', 'large']}

        for m in metadata_list:
            shape_counts[m['shape']] += 1
            size_counts[m['size_category']] += 1
                
        return {
            'counts': shape_counts,
            'size_counts': size_counts,
        }

def get_loss_weights(epoch: int) -> Dict[str, float]:
    """Get loss weights based on curriculum schedule.

    Note: Losses are now summed (not per-token averaged), so token counts
    naturally affect the loss magnitude. We use modest weights for balance.
    """
    if epoch < 3:  # Epochs 0-2: Learn basic structure
        return {'rationale': 1.0, 'answer': 2.0, 'aux': 0.5}
    elif epoch < 6:  # Epochs 3-5: Balance with emphasis on answers
        return {'rationale': 0.8, 'answer': 3.0, 'aux': 0.3}
    else:  # Epochs 6+: Strong focus on correct answers
        return {'rationale': 0.5, 'answer': 4.0, 'aux': 0.2}


def compute_weighted_loss(
    logits, target_tokens, rat_mask, ans_mask,
    aux_outputs, aux_labels, tokenizer, loss_weights
):
    V = tokenizer.get_vocab_size()
    # CE per token
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        target_tokens.reshape(-1),
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.1,
        reduction='none'
    ).view(target_tokens.size(0), -1)

    # masks: [B,T] float {0,1}
    def masked_mean(x, m):
        denom = m.sum().clamp_min(1.0)
        return (x * m).sum() / denom

    # components
    rationale_loss = masked_mean(ce, rat_mask)
    answer_loss    = masked_mean(ce, ans_mask)

    # aux (counts)
    aux_loss = 0.0

    # Shape count losses
    for shape, head_logits in aux_outputs['count_logits'].items():
        targets = torch.tensor([al['counts'][shape] for al in aux_labels],
                                device=head_logits.device, dtype=torch.long)
        aux_loss += F.cross_entropy(head_logits, targets)

    # Size count losses
    for size, head_logits in aux_outputs['size_count_logits'].items():
        targets = torch.tensor([al['size_counts'][size] for al in aux_labels],
                                device=head_logits.device, dtype=torch.long)
        aux_loss += F.cross_entropy(head_logits, targets)

    # Normalize by number of heads that contributed
    aux_loss /= (len(aux_outputs['count_logits']) + len(aux_outputs['size_count_logits']))

    total = (loss_weights['rationale'] * rationale_loss +
             loss_weights['answer']    * answer_loss +
             loss_weights['aux']       * aux_loss)

    return total, rationale_loss, answer_loss, aux_loss


def create_curriculum_datasets(text_processor, samples_per_epoch=10000):
    """Create datasets for curriculum learning with increasing difficulty."""
    datasets = []

    # Epochs 0-2: Easy questions (existence, identification)
    for _ in range(3):
        datasets.append(ShapeDataset(
            num_samples=samples_per_epoch,
            text_processor=text_processor,
            difficulty='easy'
        ))

    # Epochs 3-5: Medium questions (counting, single-hop)
    for _ in range(3):
        datasets.append(ShapeDataset(
            num_samples=samples_per_epoch,
            text_processor=text_processor,
            difficulty='medium'
        ))

    # Epochs 6-9: Hard questions (comparison, multi-hop)
    for _ in range(4):
        datasets.append(ShapeDataset(
            num_samples=samples_per_epoch,
            text_processor=text_processor,
            difficulty='hard'
        ))

    return datasets

def collate_fn(batch):
    return {
        'image':         torch.stack([b['image'] for b in batch]),
        'input_tokens':  torch.stack([b['input_tokens'] for b in batch]),
        'target_tokens': torch.stack([b['target_tokens'] for b in batch]),
        'rat_mask':      torch.stack([b['rat_mask'] for b in batch]),
        'ans_mask':      torch.stack([b['ans_mask'] for b in batch]),
        'aux_labels':    [b['aux_labels'] for b in batch],
        'metadata':      [b['metadata'] for b in batch],
        'question':      [b['question'] for b in batch],
        'answer':        [b['answer'] for b in batch],
        'rationale':     [b['rationale'] for b in batch],
    }

def train_with_curriculum(model, datasets, num_epochs=NUM_EPOCHS):
    """Train model with curriculum learning across different difficulty levels."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    print(f"Training on {DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Using curriculum learning with difficulty progression")

    for epoch in range(num_epochs):
        # Get dataset for this epoch
        dataset = datasets[epoch]
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)

        model.train()
        total_loss = 0
        total_rationale_loss = 0
        total_answer_loss = 0
        total_aux_loss = 0

        # Get loss weights for this epoch
        loss_weights = get_loss_weights(epoch)

        difficulty = dataset.difficulty
        print(f"\nEpoch {epoch+1}/{num_epochs} - Difficulty: {difficulty}")
        print(f"Loss weights: {loss_weights}")
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        for batch in progress_bar:
            images = batch['image'].to(DEVICE)
            inp = batch['input_tokens'].to(DEVICE)
            tgt = batch['target_tokens'].to(DEVICE)
            rat_mask  = batch['rat_mask'].to(DEVICE)
            ans_mask  = batch['ans_mask'].to(DEVICE)
            aux_labels = batch['aux_labels']

            logits, aux_outputs = model(images, inp, return_aux=True)

            loss, rat_loss, ans_loss, aux_loss = compute_weighted_loss(
                logits, tgt, rat_mask, ans_mask,
                aux_outputs, aux_labels, model.text_processor.tokenizer, loss_weights
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_rationale_loss += rat_loss.item()
            total_answer_loss += ans_loss.item()
            total_aux_loss += aux_loss if isinstance(aux_loss, float) else aux_loss.item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rat': f'{rat_loss.item():.3f}',
                'ans': f'{ans_loss.item():.3f}',
                'aux': f'{aux_loss if isinstance(aux_loss, float) else aux_loss.item():.3f}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_rat_loss = total_rationale_loss / len(train_loader)
        avg_ans_loss = total_answer_loss / len(train_loader)
        avg_aux_loss = total_aux_loss / len(train_loader)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Rationale Loss: {avg_rat_loss:.4f}")
        print(f"  Answer Loss: {avg_ans_loss:.4f}")
        print(f"  Aux Loss: {avg_aux_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()

    return model


def main():
    """Main training function with chain-of-thought reasoning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--samples_per_epoch", type=int, default=10000)
    args = parser.parse_args()

    print("Initializing Toy VLM with Chain-of-Thought Reasoning...")

    # Build tokenizer vocabulary from rationales (NEW - uses RationaleGenerator)
    print("Building tokenizer vocabulary from RationaleGenerator...")
    rationale_gen = RationaleGenerator()
    text_processor = TextProcessor()
    text_processor.tokenizer.build_vocab_from_rationales(rationale_gen, num_samples=200)

    # Save the tokenizer vocabulary
    text_processor.tokenizer.save_vocab('tokenizer_vocab.json')
    print(f"Vocabulary size: {text_processor.tokenizer.get_vocab_size()}")

    # Create model with the built tokenizer
    model = ToyVLM(text_processor, num_layers=6)  # Increased layers for reasoning

    # Create curriculum datasets
    print("\nCreating curriculum datasets...")
    datasets = create_curriculum_datasets(text_processor, samples_per_epoch=args.samples_per_epoch)

    # Train model with curriculum
    model = train_with_curriculum(model, datasets, num_epochs=args.epochs)

    # Save model
    torch.save(model.state_dict(), 'toy_vlm_cot.pth')
    print("\nTraining complete. Model saved as 'toy_vlm_cot.pth'")


if __name__ == "__main__":
    main()