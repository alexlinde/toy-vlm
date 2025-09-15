"""
Model components for the Toy VLM.
Contains all neural network architectures and model-related functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from text import MAX_SEQ_LEN, TextProcessor

# Model constants - device fallback
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4

class SimpleVisionEncoder(nn.Module):
    """Simple CNN for encoding images."""
    
    def __init__(self, output_dim=HIDDEN_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        
        # Calculate flattened size: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.flatten_size = 64 * 8 * 8
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output

class TransformerBlock(nn.Module):
    """Transformer decoder block."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

class ToyVLM(nn.Module):
    """Simple Vision-Language Model."""
    
    def __init__(self, text_processor=None, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        
        # Initialize text processor if not provided
        if text_processor is None:
            text_processor = TextProcessor()
        self.text_processor = text_processor
        vocab_size = text_processor.tokenizer.get_vocab_size()
        
        # Vision encoder
        self.vision_encoder = SimpleVisionEncoder(hidden_dim)
        
        # Text embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, hidden_dim)
        
        # Vision-language fusion
        self.vision_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Transformer decoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def create_causal_mask(self, seq_len, device):
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
    
    def forward(self, images, input_tokens):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device
        
        # Encode vision
        vision_features = self.vision_encoder(images)  # [batch, hidden_dim]
        vision_features = self.vision_projection(vision_features)
        vision_features = vision_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Embed tokens
        token_embeds = self.token_embedding(input_tokens)  # [batch, seq_len, hidden_dim]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        
        text_embeds = self.dropout(token_embeds + position_embeds)
        
        # Concatenate vision features as prefix
        combined = torch.cat([vision_features, text_embeds], dim=1)  # [batch, 1+seq_len, hidden_dim]
        
        # Create causal mask (including vision prefix)
        total_len = 1 + seq_len
        mask = self.create_causal_mask(total_len, device)
        
        # Pass through transformer
        hidden = combined
        for block in self.transformer_blocks:
            hidden = block(hidden, mask)
        
        # Get only text positions for output (skip vision prefix)
        text_hidden = hidden[:, 1:, :]  # [batch, seq_len, hidden_dim]
        
        # Project to vocabulary
        logits = self.output_projection(text_hidden)
        
        return logits

@torch.no_grad()
def generate_response(model, image, question, max_length=30):
    """Generate response for a given image and question."""
    model.eval()
    device = next(model.parameters()).device
    tokenizer = model.text_processor.tokenizer
    
    # Prepare image
    if isinstance(image, np.ndarray):
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    else:
        image = image.to(device)
    
    # Tokenize question
    q_tokens = tokenizer.tokenize(question)
    input_tokens = [tokenizer.bos_token_id] + q_tokens
    
    # Generate token by token
    for _ in range(max_length):
        # Prepare input tensor
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Pad if necessary
        if input_tensor.size(1) < MAX_SEQ_LEN:
            padding = torch.full((1, MAX_SEQ_LEN - input_tensor.size(1)), tokenizer.pad_token_id, dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, padding], dim=1)
        
        # Get prediction
        logits = model(image, input_tensor)
        next_token_logits = logits[0, len(input_tokens) - 1, :]
        
        # Sample next token (greedy decoding)
        next_token = torch.argmax(next_token_logits).item()
        
        if next_token == tokenizer.eos_token_id or next_token == tokenizer.pad_token_id:
            break
            
        input_tokens.append(next_token)
    
    # Decode response (skip START token and question tokens)
    response_tokens = input_tokens[len(q_tokens) + 1:]
    response = tokenizer.decode(response_tokens)
    
    return response