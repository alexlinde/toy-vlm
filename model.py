"""
Model components for the Toy VLM.
Contains all neural network architectures and model-related functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from text import MAX_SEQ_LEN
from shapes import IMAGE_SIZE

# Model constants - device fallback
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4

import torch
import torch.nn as nn

class SimpleViTEncoder(nn.Module):
    def __init__(self, d_model=HIDDEN_DIM, patch_size=8, image_size=IMAGE_SIZE):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            1, d_model, kernel_size=patch_size, stride=patch_size
        )  # (B, d_model, 8, 8)

        num_patches = (image_size // patch_size) ** 2  # 64
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.patch_embed(x)                        # (B, d_model, 8, 8)
        x = x.flatten(2).transpose(1, 2)               # (B, 64, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1) # (B, 1, d_model)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.norm(x)
        return x  # (B, 65, d_model)

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

class CrossAttention(nn.Module):
    """Cross-attention mechanism for text attending to vision features."""

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

    def forward(self, query, memory):
        batch_size = query.size(0)

        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(memory).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(memory).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention (no masking for cross-attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(context)
        return output

class TransformerBlock(nn.Module):
    """Transformer decoder block with cross-attention to vision features."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = CrossAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x, vision_memory, mask=None):
        # Self-attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)

        # Cross-attention to vision features
        cross_attn_output = self.cross_attention(x, vision_memory)
        x = self.norm2(x + cross_attn_output)

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x

class ToyVLM(nn.Module):
    """Simple Vision-Language Model."""
    
    def __init__(self, text_processor, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        
        # Text processor
        self.text_processor = text_processor
        vocab_size = text_processor.tokenizer.get_vocab_size()
        
        # Vision encoder (ViT-style with shared d_model and positional scheme)
        self.vision_encoder = SimpleViTEncoder(d_model=hidden_dim)
        
        # Text embeddings (share same d_model and simple 1D learned positions)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, hidden_dim)
        
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

        # Encode vision features as memory tokens (CLS + patches) already positioned and normalized
        vision_memory = self.vision_encoder(images)  # [batch, 65, hidden_dim]

        # Embed text tokens
        token_embeds = self.token_embedding(input_tokens)  # [batch, seq_len, hidden_dim]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)

        text_embeds = self.dropout(token_embeds + position_embeds)

        # Create causal mask for text only
        mask = self.create_causal_mask(seq_len, device)

        # Pass through transformer with cross-attention
        hidden = text_embeds
        for block in self.transformer_blocks:
            hidden = block(hidden, vision_memory, mask)

        # Project to vocabulary
        logits = self.output_projection(hidden)

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