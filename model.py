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
    """CNN encoder that produces 8x8 spatial tokens."""

    def __init__(self, output_dim=HIDDEN_DIM):
        super().__init__()
        self.output_dim = output_dim

        # Input: (B, num_channels, 64, 64)
        # Conv layers to downsample to 8x8
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # -> (B, 32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # -> (B, 64, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)  # -> (B, 128, 8, 8)

        # Project each 8x8 spatial location to output_dim
        self.projection = nn.Conv2d(128, output_dim, kernel_size=1)

        # Learnable 2D positional embeddings for 8x8 grid
        self.pos_embed = nn.Parameter(torch.randn(1, output_dim, 8, 8) * 0.02)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, C, 64, 64) where C is num_channels
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # (B, 128, 8, 8)

        # Project to output dimension
        x = self.projection(x)  # (B, output_dim, 8, 8)

        # Add positional embeddings
        x = x + self.pos_embed

        x = self.dropout(x)

        # Flatten spatial dimensions: (B, output_dim, 8, 8) -> (B, 64, output_dim)
        B = x.size(0)
        x = x.view(B, self.output_dim, 64).transpose(1, 2)  # (B, 64, output_dim)

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
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        
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

class AuxiliaryHeads(nn.Module):
    """Auxiliary prediction heads for vision tokens."""

    def __init__(self, hidden_dim, num_shapes=5, num_colors=3, num_sizes=3):
        super().__init__()

        # Per-token heads (applied to each of 64 spatial tokens)
        self.shape_classifier = nn.Linear(hidden_dim, num_shapes + 1)  # 5 shapes + background
        self.color_classifier = nn.Linear(hidden_dim, num_colors)  # red, green, blue
        self.size_classifier = nn.Linear(hidden_dim, num_sizes)  # small, medium, large

        # Global heads (applied to pooled representation)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Count predictor: for each shape type, predict 0-4
        self.count_heads = nn.ModuleDict({
            shape: nn.Linear(hidden_dim, 5) for shape in
            ['square', 'circle', 'rectangle', 'cross', 'triangle']
        })

    def forward(self, vision_tokens):
        """
        Args:
            vision_tokens: [batch, 64, hidden_dim]

        Returns:
            dict with keys: 'shape_logits', 'color_logits', 'size_logits', 'count_logits'
        """
        # Per-token predictions
        shape_logits = self.shape_classifier(vision_tokens)  # [B, 64, 6]
        color_logits = self.color_classifier(vision_tokens)  # [B, 64, 3]
        size_logits = self.size_classifier(vision_tokens)  # [B, 64, 3]

        # Global pooling for count prediction
        pooled = self.global_pool(vision_tokens.transpose(1, 2)).squeeze(-1)  # [B, hidden_dim]

        # Count predictions for each shape type
        count_logits = {}
        for shape, head in self.count_heads.items():
            count_logits[shape] = head(pooled)  # [B, 5] (0-4 counts)

        return {
            'shape_logits': shape_logits,
            'color_logits': color_logits,
            'size_logits': size_logits,
            'count_logits': count_logits
        }


class ToyVLM(nn.Module):
    """Simple Vision-Language Model with Chain-of-Thought reasoning."""

    def __init__(self, text_processor=None, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()

        # Initialize text processor if not provided
        if text_processor is None:
            text_processor = TextProcessor()
        self.text_processor = text_processor
        vocab_size = text_processor.tokenizer.get_vocab_size()

        # Vision encoder (now produces 64 tokens)
        self.vision_encoder = SimpleVisionEncoder(HIDDEN_DIM)

        # Auxiliary heads for multi-task learning
        self.aux_heads = AuxiliaryHeads(hidden_dim)

        # Text embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, hidden_dim)

        # Type embeddings for text and vision
        self.text_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.vision_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Vision-language fusion
        self.vision_projection = nn.Linear(hidden_dim, hidden_dim)

        # Transformer decoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # Output projection with weight tying
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.output_projection.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(0.1)

        # Xavier initialization for linear and conv layers
        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init_weights)
        
    def create_causal_mask(self, seq_len, device):
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
    
    def encode_image(self, images):
        """Encode image to vision memory. Can be cached for generation."""
        vision_tokens = self.vision_encoder(images)  # [batch, 64, hidden_dim]
        vision_memory = self.vision_projection(vision_tokens)  # [batch, 64, hidden_dim]
        return vision_memory + self.vision_type  # Add vision type embedding

    def forward_with_memory(self, vision_memory, input_tokens):
        """Forward pass with pre-encoded vision memory (for efficient generation)."""
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        # Embed text tokens with type embedding
        token_embeds = self.token_embedding(input_tokens)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        text_embeds = self.dropout(token_embeds + self.position_embedding(positions) + self.text_type)

        # Create causal mask for text only
        mask = self.create_causal_mask(seq_len, device)

        # Pass through transformer with cross-attention to vision memory
        hidden = text_embeds
        for block in self.transformer_blocks:
            hidden = block(hidden, vision_memory, mask)

        # Project to vocabulary
        return self.output_projection(hidden)

    def forward(self, images, input_tokens, return_aux=False):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        # Encode vision features as memory: [batch, 64, hidden_dim]
        vision_tokens = self.vision_encoder(images)  # [batch, 64, hidden_dim]

        # Compute auxiliary predictions
        aux_outputs = None
        if return_aux:
            aux_outputs = self.aux_heads(vision_tokens)

        vision_memory = self.vision_projection(vision_tokens)  # [batch, 64, hidden_dim]
        vision_memory = vision_memory + self.vision_type  # Add vision type embedding

        # Embed text tokens with type embedding
        token_embeds = self.token_embedding(input_tokens)  # [batch, seq_len, hidden_dim]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)

        text_embeds = self.dropout(token_embeds + position_embeds + self.text_type)

        # Create causal mask for text only
        mask = self.create_causal_mask(seq_len, device)

        # Pass through transformer with cross-attention to vision memory
        hidden = text_embeds
        for block in self.transformer_blocks:
            hidden = block(hidden, vision_memory, mask)

        # Project to vocabulary
        logits = self.output_projection(hidden)

        if return_aux:
            return logits, aux_outputs
        return logits

def _find_first(seq, token_id):
    for i,t in enumerate(seq):
        if t == token_id: return i
    return None

def _sample_logits(logits, temperature=0.7):
    """Sample from logits with temperature."""
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()

@torch.no_grad()
def generate_response(model, image, question, max_length=35, return_rationale=True):
    """Generate response with optional chain-of-thought rationale.

    Format: <START> question <Q> <REASON> rationale <SEP> <FINAL> answer

    Args:
        model: ToyVLM model
        image: Input image tensor (C, H, W)
        question: Question string
        max_length: Maximum generation length per stage
        return_rationale: If True, return (rationale, answer). If False, return answer only.

    Returns:
        If return_rationale: (rationale_str, answer_str)
        Otherwise: answer_str
    """
    model.eval()
    device = next(model.parameters()).device
    tokenizer = model.text_processor.tokenizer

    # Prepare image: (C, H, W) -> (1, C, H, W)
    image = image.unsqueeze(0).to(device)

    # Encode image once
    vision_memory = model.encode_image(image)

    # Tokenize question and add special tokens
    # We generate everything after <Q>, so stop there
    q_tokens = tokenizer.tokenize(question)
    input_tokens = [tokenizer.bos_token_id] + q_tokens + [tokenizer.q_token_id]

    # Track where generation starts (after <Q>)
    generation_start_idx = len(input_tokens)

    # 1) Generate rationale until <SEP>
    for _ in range(max_length):
        # Prepare input tensor
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

        # Pad if necessary
        if input_tensor.size(1) < MAX_SEQ_LEN:
            padding = torch.full((1, MAX_SEQ_LEN - input_tensor.size(1)), tokenizer.pad_token_id, dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, padding], dim=1)
        elif input_tensor.size(1) > MAX_SEQ_LEN:
            # Truncate if too long
            input_tensor = input_tensor[:, :MAX_SEQ_LEN]

        # Get prediction using cached vision memory
        logits = model.forward_with_memory(vision_memory, input_tensor)
        next_token_logits = logits[0, min(len(input_tokens) - 1, MAX_SEQ_LEN - 1), :]

        # Sample next token with temperature (stochastic for rationale)
        next_token = _sample_logits(next_token_logits, temperature=0.7)

        # Check for end of rationale generation
        if next_token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break

        input_tokens.append(next_token)

        if next_token == tokenizer.sep_token_id:
            break

    # Force <FINAL> to start the answer span
    input_tokens.append(tokenizer.final_token_id)

    # Detect answer type for constrained decoding
    question_lower = question.lower()
    expects_yes_no = any(q in question_lower for q in ['is there', 'are there more', 'does it'])
    expects_digit = any(q in question_lower for q in ['how many'])

    # Build allowed token sets for constrained decoding
    allowed_tokens = None
    if expects_yes_no:
        allowed_tokens = []
        for word in ['yes', 'no', 'equal', 'they', 'are']:
            if word in tokenizer.vocab:
                allowed_tokens.append(tokenizer.vocab[word])
        # Always allow end tokens
        allowed_tokens.extend([tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id])
        allowed_tokens = torch.tensor(allowed_tokens, device=device) if allowed_tokens else None
    elif expects_digit:
        allowed_tokens = []
        for digit in ['zero', 'one', 'two', 'three', 'four', '0', '1', '2', '3', '4']:
            if digit in tokenizer.vocab:
                allowed_tokens.append(tokenizer.vocab[digit])
        # Always allow end tokens
        allowed_tokens.extend([tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id])
        allowed_tokens = torch.tensor(allowed_tokens, device=device) if allowed_tokens else None

    # 2) Generate answer (optionally use constrained vocab)
    for _ in range(max_length):
        # Prepare input tensor
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

        # Pad if necessary
        if input_tensor.size(1) < MAX_SEQ_LEN:
            padding = torch.full((1, MAX_SEQ_LEN - input_tensor.size(1)), tokenizer.pad_token_id, dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, padding], dim=1)
        elif input_tensor.size(1) > MAX_SEQ_LEN:
            # Truncate if too long
            input_tensor = input_tensor[:, :MAX_SEQ_LEN]

        # Get prediction using cached vision memory
        logits = model.forward_with_memory(vision_memory, input_tensor)
        next_token_logits = logits[0, min(len(input_tokens) - 1, MAX_SEQ_LEN - 1), :]

        # Constrained decoding for structured answers
        if allowed_tokens is not None:
            constrained_logits = next_token_logits[allowed_tokens]
            next_token_idx = torch.argmax(constrained_logits).item()
            next_token = allowed_tokens[next_token_idx].item()
        else:
            next_token = torch.argmax(next_token_logits).item()

        # Check for end of answer generation
        if next_token in (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id):
            break

        input_tokens.append(next_token)

    # Parse spans safely
    # Generated sequence: <REASON> rationale <SEP> <FINAL> answer <END>
    reason_idx = _find_first(input_tokens, tokenizer.reason_token_id)
    sep_idx = _find_first(input_tokens, tokenizer.sep_token_id)
    final_idx = _find_first(input_tokens, tokenizer.final_token_id)

    rationale_tokens, answer_tokens = [], []
    if reason_idx is not None and sep_idx is not None and sep_idx > reason_idx:
        # Rationale is between <REASON> and <SEP>
        rationale_tokens = input_tokens[reason_idx+1:sep_idx]

    if final_idx is not None and final_idx > (sep_idx or 0):
        # Answer is after <FINAL>
        answer_tokens = input_tokens[final_idx+1:]
        # Remove any end tokens
        answer_tokens = [t for t in answer_tokens if t not in (tokenizer.eos_token_id, tokenizer.pad_token_id)]

    rationale = tokenizer.decode(rationale_tokens, skip_special_tokens=True)
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return (rationale, answer) if return_rationale else answer