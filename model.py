"""
Model components for the Toy VLM.
Contains all neural network architectures and model-related functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from text import MAX_SEQ_LEN, NUM_IMG_TOKENS, TextProcessor
from shapes import ObjType, ObjSize

# Model constants - device fallback
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 6

# Squeeze-and-Excitation (SE) module
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(1, c//r))
        self.fc2 = nn.Linear(max(1, c//r), c)
    def forward(self, x):  # x: (B,C,H,W)
        b,c,h,w = x.shape
        s = x.mean(dim=(2,3))                 # (B,C)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))        # (B,C)
        return x * s.view(b,c,1,1)

class VisionTokenEncoder(nn.Module):
    """Tiny CNN -> 8x8 grid tokens -> linear to hidden_dim, with ALWAYS-ON (x,y) coord channels."""

    def __init__(self, hidden_dim: int, channels: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channels = channels

        # Input: (B, 1, 64, 64) - grayscale images; we'll concat (x,y) -> (B, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(2)  # -> (B, 16, 32, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(2)  # -> (B, 32, 16, 16)
        self.conv3 = nn.Conv2d(32, channels, kernel_size=3, padding=1, groups=4)
        self.pool3 = nn.AvgPool2d(2)  # -> (B, C, 8, 8)

        # Learnable 2D positional embeddings aligned to the 8x8 conv feature map
        self.pos_embed_2d = nn.Parameter(torch.randn(1, channels, 8, 8) * 0.02)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))

        # Linear projection per-token to hidden_dim
        self.proj = nn.Linear(channels, hidden_dim)
        self.drop = nn.Dropout(p=0.1)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-5)

        self.se3 = SE(channels)        

    @staticmethod
    def _coord_channels(B: int, H: int, W: int, device, dtype):
        """
        Returns (B, 2, H, W): x in [0,1] increasing left->right, y in [0,1] top->bottom.
        """
        y = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        x = torch.linspace(0.0, 1.0, steps=W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        # x: (B, 1, 64, 64)
        B, C, H, W = x.shape
        assert C == 1, f"Expected grayscale input with 1 channel, got {C}"
        assert H % 8 == 0 and W % 8 == 0, "Image size should be divisible by 8 for 8x8 grid downstream."

        # ALWAYS add (x,y) coordinate channels
        coords = self._coord_channels(B, H, W, x.device, x.dtype)  # (B, 2, H, W)
        x = torch.cat([x, coords], dim=1)  # (B, 3, H, W)

        # Tiny CNN stem to 8x8 feature grid
        x = F.silu(self.conv1(x)); x = self.pool1(x)
        x = F.silu(self.conv2(x)); x = self.pool2(x)
        x = F.silu(self.conv3(x)); x = self.se3(x); x = self.pool3(x)  # (B, C, 8, 8)

        # Add 2D positional encoding at the 8x8 stage
        x = x + self.pos_scale * self.pos_embed_2d

        # Optional downsample to match NUM_IMG_TOKENS grid, then flatten to tokens
        B, C, H, W = x.shape  # expected 8x8
        if H * W != NUM_IMG_TOKENS:
            G = int(math.sqrt(NUM_IMG_TOKENS))
            assert G * G == NUM_IMG_TOKENS, f"NUM_IMG_TOKENS must be a perfect square, got {NUM_IMG_TOKENS}"
            assert H % G == 0 and W % G == 0, f"Cannot pool from {H}x{W} to {G}x{G}"
            k_h = H // G
            k_w = W // G
            x = F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w))  # (B, C, G, G)
            H, W = G, G

        # Flatten to tokens -> (B, NUM_IMG_TOKENS, C)
        tokens = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # Project to hidden_dim and layer-norm
        tokens = self.proj(tokens)                       # (B, N, hidden_dim)
        tokens = self.drop(self.proj(tokens))
        tokens = self.ln(tokens)
        
        # NaN/Inf guard
        assert torch.isfinite(tokens).all(), "[model] NaN/Inf in vision tokens"
        return tokens

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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class AuxiliaryHeads(nn.Module):
    """Auxiliary heads for global object and size counts.

    These operate on a pooled representation of the visual tokens. We keep them
    lightweight: a small MLP that predicts capped counts (0..4) for each shape
    type and for each size category.
    """

    def __init__(self, hidden_dim: int, max_count: int = 5):
        super().__init__()
        # Predict counts in {0,1,2,3,4} -> num_classes = 5
        self.num_classes = max_count
        # Number of shapes and sizes
        self.num_shapes = len(ObjType)
        self.num_sizes = len(ObjSize)

        # Shared pooling MLP over token dimension
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-task linear heads
        self.shape_count_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_classes) for _ in range(self.num_shapes)
        ])
        self.size_count_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_classes) for _ in range(self.num_sizes)
        ])

    def forward(self, vision_tokens: torch.Tensor):
        """Compute auxiliary predictions.

        Args:
            vision_tokens: Tensor [B, N, H] pooled image tokens used as visual prefix
        Returns:
            dict with keys:
              - 'count_logits': {shape_name: Tensor[B, 5]}
              - 'size_count_logits': {size_name: Tensor[B, 5]}
        """
        assert vision_tokens is not None, "vision_tokens must be provided to aux heads"
        # Mean pool across tokens -> [B, H]
        pooled = vision_tokens.mean(dim=1)
        pooled = self.pool_norm(pooled)
        pooled = self.pool_proj(pooled)

        # Shape counts
        count_logits = {}
        for idx, obj in enumerate(ObjType):
            head = self.shape_count_heads[idx]
            count_logits[obj.value] = head(pooled)

        # Size counts
        size_count_logits = {}
        for idx, size in enumerate(ObjSize):
            head = self.size_count_heads[idx]
            size_count_logits[size.value] = head(pooled)

        return {
            'count_logits': count_logits,
            'size_count_logits': size_count_logits,
        }


class ToyVLM(nn.Module):
    """Vision-Language Model that prefixes image tokens into the text sequence."""

    def __init__(self, text_processor):
        super().__init__()

        # Vision token encoder (produces 8x8 -> 64 tokens at hidden_dim)
        self.vision_token_encoder = VisionTokenEncoder(HIDDEN_DIM)

        self.text_processor = text_processor
        vocab_size = text_processor.tokenizer.get_vocab_size()

        # Text embeddings
        self.token_embedding = nn.Embedding(vocab_size, HIDDEN_DIM)
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, HIDDEN_DIM)

        # Type embeddings for text and vision tokens
        self.text_type = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.vision_type = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.input_norm = nn.LayerNorm(HIDDEN_DIM)

        # Transformer decoder (self-attention only)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(HIDDEN_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)
        ])
        self.dropout = nn.Dropout(0.1)

        # Xavier initialization for linear and conv layers
        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # build modules
        self.output_projection = nn.Linear(HIDDEN_DIM, vocab_size)
        self.auxiliary_heads = AuxiliaryHeads(HIDDEN_DIM)

        # init *before* tying
        self.apply(_init_weights)  # only touches Linear/Conv, not Embedding

        # now tie (so Embedding keeps its intended init)
        self.output_projection.weight = self.token_embedding.weight
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)  # common practice

    def create_causal_mask(self, seq_len, device):
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
    
    def encode_image_tokens(self, images):
        """Encode image into exactly NUM_IMG_TOKENS visual tokens [B, N, H]."""
        return self.vision_token_encoder(images)

    def forward_with_embeds(self, input_embeds):
        batch_size, seq_len, _ = input_embeds.shape
        device = input_embeds.device
        mask = self.create_causal_mask(seq_len, device)
        hidden = input_embeds
        for block in self.transformer_blocks:
            hidden = block(hidden, mask)
        return self.output_projection(hidden)

    def forward(self, images, input_tokens, return_aux=False):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        # 1) Encode image into a fixed set of tokens
        img_tokens = self.encode_image_tokens(images)  # [B, 64, H] for 8x8 grid

        # 2) Embed input token IDs
        token_embeds = self.token_embedding(input_tokens)  # [B, T, H]

        # 3) Replace <IMG> placeholder embeddings with visual tokens in order
        tok = self.text_processor.tokenizer
        for b in range(batch_size):
            img_positions = (input_tokens[b] == tok.img_token_id).nonzero(as_tuple=True)[0]
            n = img_positions.numel()
            assert n == NUM_IMG_TOKENS, f"expected {NUM_IMG_TOKENS} image tokens, got {n}"
            btok = img_tokens[b:b+1]  # [1, NUM_IMG_TOKENS, H]
            token_embeds[b, img_positions, :] = btok[0]
            # Add vision type embedding to image token positions
            token_embeds[b, img_positions, :] = token_embeds[b, img_positions, :] + self.vision_type

        # Basic structure checks
        assert torch.isfinite(token_embeds).all(), "[model] NaN/Inf in token_embeds"

        # 4) Add positional and type embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        input_embeds = token_embeds + self.position_embedding(positions) + self.text_type
        input_embeds = self.input_norm(input_embeds)
        input_embeds = self.dropout(input_embeds)

        # 5) Decode with self-attention only
        logits = self.forward_with_embeds(input_embeds)
        assert torch.isfinite(logits).all(), "[model] NaN/Inf in logits"
        aux_outputs = None
        if return_aux:
            # Provide auxiliary heads with the unmodified visual tokens
            aux_outputs = self.auxiliary_heads(img_tokens)

        return logits if not return_aux else (logits, aux_outputs)

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
    """Generate response with optional chain-of-thought rationale using new prompt.

    New format:
      [BOS] [|user|] q + <IMG_START> <IMG>xN <IMG_END> [|assistant|] <THINK>
      rationale </THINK> <FINAL> answer </FINAL> <EOS>

    Args:
        model: ToyVLM instance
        image: Input image tensor (C, H, W) in [0,1] or [0,255]
        question: Question string
        max_length: Maximum steps per stage (rationale/answer)
        return_rationale: If True returns (rationale, answer); else answer
    """
    model.eval()
    device = next(model.parameters()).device
    tok = model.text_processor.tokenizer

    # Image must already be the correct format: (1, 64, 64), float in [0,1]
    assert isinstance(image, torch.Tensor), "image must be a torch.Tensor"
    assert image.ndim == 3 and image.size(0) == 1 and image.size(1) == 64 and image.size(2) == 64, \
        f"expected image shape (1,64,64), got {tuple(image.shape)}"
    assert image.dtype == torch.float32, f"expected image dtype float32, got {image.dtype}"
    imin = float(image.min())
    imax = float(image.max())
    assert 0.0 <= imin and imax <= 1.0, f"expected image normalized to [0,1], got range [{imin:.3f}, {imax:.3f}]"
    image = image.to(device)

    # Build prompt tokens
    # Use allow_unk=True for free-form questions
    q_ids = tok.tokenize(question, allow_unk=True)
    num_img_tokens = NUM_IMG_TOKENS
    img_block = [tok.img_start_id] + [tok.img_token_id] * num_img_tokens + [tok.img_end_id]
    input_ids = [tok.bos_token_id] + [tok.user_token_id] + q_ids + img_block + [tok.assistant_token_id] + [tok.think_start_id]

    rationale_tokens = []
    answer_tokens = []

    def step_once(ids, allowed=None, end_id=None, prefer_after=None):
        # Pad sequence and run forward to get next-token logits at last position
        ids_pad = model.text_processor.pad_sequence(ids, MAX_SEQ_LEN)
        ids_t = torch.tensor(ids_pad, dtype=torch.long, device=device).unsqueeze(0)
        img_t = image.unsqueeze(0)
        logits = model(img_t, ids_t)
        # Use the last real token position (before padding) for next-token logits
        pos = min(len(ids) - 1, MAX_SEQ_LEN - 1)
        next_logits = logits[0, pos, :].clone()
        if allowed:
            idx = torch.tensor(allowed, device=next_logits.device, dtype=torch.long)
            sub = next_logits[idx]
            # Encourage ending token after minimal content
            if end_id is not None and prefer_after is not None and (len(ids) >= prefer_after):
                for j, tid in enumerate(allowed):
                    if tid == end_id:
                        sub[j] = sub[j] + 1.0
            choice = int(idx[int(torch.argmax(sub)).item()].item())
            return choice
        return int(torch.argmax(next_logits).item())

    # Rationale stage: generate until </THINK> or limit
    for _ in range(max_length):
        nxt = step_once(input_ids, allowed=None, end_id=tok.think_end_id, prefer_after=2)
        if nxt in (tok.eos_token_id, tok.pad_token_id):
            break
        input_ids.append(nxt)
        if nxt == tok.think_end_id:
            break
        rationale_tokens.append(nxt)
    # Ensure THINK is closed
    if not (len(input_ids) and input_ids[-1] == tok.think_end_id):
        input_ids.append(tok.think_end_id)

    # Append <FINAL>
    input_ids.append(tok.final_start_id)

    # Answer stage: generate until </FINAL> or single yes/no token
    for _ in range(max_length):
        nxt = step_once(input_ids, allowed=None, end_id=tok.final_end_id, prefer_after=1)
        if nxt in (tok.eos_token_id, tok.pad_token_id):
            break
        if nxt == tok.final_end_id:
            input_ids.append(nxt)
            break
        answer_tokens.append(nxt)
        input_ids.append(nxt)

    # Ensure closing </FINAL> and <EOS>
    if not (len(input_ids) and input_ids[-1] == tok.final_end_id):
        input_ids.append(tok.final_end_id)
    if not (len(input_ids) and input_ids[-1] == tok.eos_token_id):
        input_ids.append(tok.eos_token_id)

    rationale = tok.decode(rationale_tokens, skip_special_tokens=True)
    answer = tok.decode(answer_tokens, skip_special_tokens=True)
    return (rationale, answer) if return_rationale else answer