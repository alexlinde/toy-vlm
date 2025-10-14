"""
Toy Vision Language Model (VLM) in PyTorch
A simple VLM that can understand basic shapes and answer questions about them.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from shapes import ShapeGenerator
from questions import QuestionGenerator
from text import TextProcessor, MAX_SEQ_LEN
from model import ToyVLM, DEVICE
import math

# Training constants
BATCH_SIZE = 64
NUM_SAMPLES = 500 * BATCH_SIZE
NUM_EPOCHS = 10
LEARNING_RATE = 4e-4
WARMUP_STEPS = 500
TOTAL_STEPS = NUM_EPOCHS * NUM_SAMPLES // BATCH_SIZE

class ShapeDataset(Dataset):
    """Dataset that generates simple geometric shapes with Q&A pairs."""
    
    def __init__(self, num_samples: int, text_processor: TextProcessor = None):
        self.num_samples = num_samples
        self.shape_generator = ShapeGenerator()
        self.question_generator = QuestionGenerator()
        self.text_processor = text_processor or TextProcessor()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random shape
        shape_type, image = self.shape_generator.generate_random_shape()
        
        # Generate Q&A pair
        question, answer = self.question_generator.generate_qa_pair(shape_type)
        
        # Prepare sequences using text processor
        input_tokens, target_tokens = self.text_processor.prepare_input_sequence(question, answer)
        
        # Pad sequences
        if len(input_tokens) > MAX_SEQ_LEN:
            input_tokens = input_tokens[:MAX_SEQ_LEN]
        if len(target_tokens) > MAX_SEQ_LEN:
            target_tokens = target_tokens[:MAX_SEQ_LEN]
            
        input_len = len(input_tokens)
        target_len = len(target_tokens)
        
        input_tokens = self.text_processor.pad_sequence(input_tokens)
        target_tokens = self.text_processor.pad_sequence(target_tokens)
        
        return {
            'image': torch.tensor(image, dtype=torch.float32).unsqueeze(0),  # Add channel dim
            'input_tokens': torch.tensor(input_tokens, dtype=torch.long),
            'target_tokens': torch.tensor(target_tokens, dtype=torch.long),
            'input_len': input_len,
            'target_len': target_len,
            'question': question,
            'answer': answer
        }

def train_model(model, train_loader, num_epochs=NUM_EPOCHS):
    """Train the VLM model."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / WARMUP_STEPS
        # cosine decay after warmup
        t = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    print(f"Training on {DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            images = batch['image'].to(DEVICE)
            input_tokens = batch['input_tokens'].to(DEVICE)
            target_tokens = batch['target_tokens'].to(DEVICE)
            
            # Forward pass
            logits = model(images, input_tokens)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, model.text_processor.tokenizer.get_vocab_size()),
                target_tokens.reshape(-1),
                ignore_index=model.text_processor.tokenizer.pad_token_id
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
        
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}")
        
    return model


def main():
    """Main training function."""
    print("Initializing Toy VLM...")
    
    # Build tokenizer vocabulary from questions
    print("Building tokenizer vocabulary...")
    question_gen = QuestionGenerator()
    text_processor = TextProcessor()
    text_processor.tokenizer.build_vocab_from_questions(question_gen)
    
    # Save the tokenizer vocabulary
    text_processor.tokenizer.save_vocab('tokenizer_vocab.json')
    
    # Create model with the built tokenizer
    model = ToyVLM(text_processor)
    
    # Create dataset with the built text processor
    train_dataset = ShapeDataset(num_samples=NUM_SAMPLES, text_processor=text_processor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train model
    model = train_model(model, train_loader, num_epochs=NUM_EPOCHS)
    
    # Save model
    torch.save(model.state_dict(), 'toy_vlm.pth')
    print("Training complete. Model and tokenizer saved.")

if __name__ == "__main__":
    main()