"""
Model components for the Toy VLM.
Contains all neural network architectures and model-related functionality.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from text import MAX_SEQ_LEN, SimpleTokenizer, TextProcessor
from shapes import IMAGE_SIZE

# Device now sourced from device.py
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
PATCH_SIZE = 8
PATCH_GRID = IMAGE_SIZE // PATCH_SIZE          # 8 patches per side
NUM_PATCHES = PATCH_GRID ** 2                  # 64 patch tokens, plus CLS = 65
PROBE_FEATURE_DIM = NUM_PATCHES * HIDDEN_DIM
# Longest trained answer is 5 words plus EOS; a question leaving fewer
# generation slots than that would silently truncate (or empty) the answer.
MAX_ANSWER_TOKENS = 6

class SimpleViTEncoder(nn.Module):
    def __init__(self, d_model=HIDDEN_DIM, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE):
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
    """Multi-head attention, used for both the self- and cross-attention slots.

    `forward(x)` attends x to itself; `forward(x, memory)` attends x to memory.
    The two differ only in where K and V come from, so one class covers both --
    the block still owns them under separate names, so each keeps its own
    weights.

    Attention capture is opt-in: set `store_attention = True` and every forward
    stashes its post-softmax weights, detached, in `last_attention`
    (B, heads, q_len, kv_len). Off by default, because building that matrix is
    the only reason not to let F.scaled_dot_product_attention fuse the whole
    thing. generate_response_traced turns it on for the cross-attention slots,
    which is what the GUI heatmap reads.
    """

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

        # Plain attributes, not buffers: nothing here belongs in a checkpoint.
        self.store_attention = False
        self.last_attention = None

    def forward(self, x, memory=None, mask=None):
        B, T, _ = x.shape
        kv = x if memory is None else memory

        def heads(proj, t):
            return proj(t).view(B, t.size(1), self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = heads(self.W_q, x), heads(self.W_k, kv), heads(self.W_v, kv)

        if self.store_attention:
            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
            if mask is not None:
                # mask: bool, True = keep. A hard-coded -1e9 would overflow to
                # -inf in fp16, so take the dtype's own floor.
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attention = F.softmax(scores, dim=-1)
            self.last_attention = attention.detach()
            context = attention @ V
        else:
            context = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

        return self.W_o(context.transpose(1, 2).reshape(B, T, self.d_model))


class TransformerBlock(nn.Module):
    """Transformer decoder block with cross-attention to vision features."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
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
        attn_output = self.self_attention(x, mask=mask)
        x = self.norm1(x + attn_output)

        # Cross-attention to vision features
        cross_attn_output = self.cross_attention(x, memory=vision_memory)
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

        # <UNK> never occurs in the exhaustively-enumerated training data, so its
        # randomly-initialized embedding would survive training as pure noise.
        with torch.no_grad():
            self.token_embedding.weight[text_processor.tokenizer.unk_token_id].zero_()

    def create_causal_mask(self, seq_len, device):
        # True = keep: lower triangle including the diagonal.
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    
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

def vision_probe_features(model, images):
    """Frozen vision features the shape probe reads: the flattened patch grid.

    Deliberately *not* the CLS token. SimpleViTEncoder has no self-attention
    layers, so nothing ever mixes patch content into CLS: its output is a
    learned constant, identical for every image, and a probe on it is provably
    stuck at chance. The patch tokens are where this encoder's view of the
    image actually lives. Returns (B, PROBE_FEATURE_DIM).
    """
    return model.vision_encoder(images)[:, 1:].flatten(1)


class ShapeProbe(nn.Module):
    """Linear probe over the frozen vision patch embeddings, classifying the shape."""

    def __init__(self, classes, d_model=PROBE_FEATURE_DIM):
        super().__init__()
        self.classes = list(classes)
        self.linear = nn.Linear(d_model, len(self.classes))
        # Temperature-scaling factor fitted on held-out data after training,
        # so displayed probabilities are calibrated rather than overconfident
        # (the raw probe reads ~100% on predictions that are right ~half the time).
        self.register_buffer('temperature', torch.ones(()))

    def forward(self, features):  # (B, d_model) -> (B, num_classes)
        return self.linear(features) / self.temperature


@torch.no_grad()
def shape_probe_probabilities(model, probe, image):
    """Classify the image with the linear probe on the frozen vision patch
    embeddings. image is a (64,64) float numpy array. Returns {class: prob}."""
    model.eval()
    device = next(model.parameters()).device
    probe = probe.to(device)  # idempotent when already there
    probe.eval()

    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    features = vision_probe_features(model, image_tensor)
    probs = F.softmax(probe(features).float(), dim=-1)[0]

    return dict(zip(probe.classes, probs.tolist()))


@torch.no_grad()
def model_shape_beliefs(model, image, shape_names):
    """The model's own belief about the shape, read from its language head.

    Asks the model 'what shape is this' with the trained answer preamble
    'this is a' teacher-forced, then reads the next-token distribution at the
    slot where the shape name goes, renormalized over shape_names. Unlike the
    linear probe (which sees only the frozen vision patch embeddings), this
    uses the full network -- cross-attention is where shape recognition
    actually happens in this architecture. Returns {shape: prob}.
    """
    model.eval()
    device = next(model.parameters()).device
    tokenizer = model.text_processor.tokenizer

    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    tokens = (
        [tokenizer.bos_token_id]
        + tokenizer.tokenize('what shape is this')
        + tokenizer.tokenize('this is a')
    )
    input_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    logits = model(image_tensor, input_tensor)[0, len(tokens) - 1, :]
    shape_ids = [tokenizer.vocab[name] for name in shape_names]
    probs = F.softmax(logits[shape_ids].float(), dim=-1)

    return dict(zip(shape_names, probs.tolist()))


def load_trained_model(checkpoint_path: str):
    """Load a trained ToyVLM, its tokenizer, and its shape probe from a checkpoint.

    Checkpoints bundle the vocabulary with the weights so the pair can never
    mismatch. Returns (model, tokenizer, probe) on CPU; the caller moves the
    model to its device. The probe is optional -- checkpoints trained before it
    existed load fine and yield probe=None.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if not (isinstance(ckpt, dict) and 'state_dict' in ckpt and 'vocab' in ckpt):
        raise ValueError(
            f"'{checkpoint_path}' is not a bundled checkpoint (expected keys "
            "'state_dict' and 'vocab'); retrain with train_model.py"
        )

    tokenizer = SimpleTokenizer.from_vocab(ckpt['vocab'])
    text_processor = TextProcessor()
    text_processor.tokenizer = tokenizer

    model = ToyVLM(text_processor)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    probe = None
    if 'probe' in ckpt and 'probe_classes' in ckpt:
        # Size the probe from its own saved weights, so a checkpoint stays
        # loadable if the feature the probe reads is ever changed.
        probe = ShapeProbe(ckpt['probe_classes'], d_model=ckpt['probe']['linear.weight'].shape[1])
        probe.load_state_dict(ckpt['probe'])
        probe.eval()

    return model, tokenizer, probe


@torch.no_grad()
def _generate(model, image, question, top_k=3, with_trace=False):
    """Greedy decode. Builds the introspection trace only when asked for it.

    Cross-attention capture costs a materialized (q_len, kv_len) matrix per
    layer per step, so it is switched on for the duration of a traced call and
    off again afterwards -- an untraced call never pays for it.
    """
    model.eval()
    device = next(model.parameters()).device
    tokenizer = model.text_processor.tokenizer

    # Prepare image tensor [1, 1, H, W]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Seed tokens with BOS + tokenized question
    q_tokens = tokenizer.tokenize(question)
    input_tokens = [tokenizer.bos_token_id] + q_tokens

    if len(input_tokens) > MAX_SEQ_LEN - MAX_ANSWER_TOKENS:
        raise ValueError(
            f"Question too long: {len(q_tokens)} words, max {MAX_SEQ_LEN - MAX_ANSWER_TOKENS - 1}"
        )

    trace = []
    for block in model.transformer_blocks:
        block.cross_attention.store_attention = with_trace

    try:
        # Autoregressive greedy decoding, capped to MAX_SEQ_LEN
        for _ in range(MAX_SEQ_LEN - len(input_tokens)):
            input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(image, input_tensor)
            pos = len(input_tokens) - 1  # position whose logits choose the next token
            next_token_logits = logits[0, pos, :]
            next_token = int(torch.argmax(next_token_logits))

            if next_token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                break

            if with_trace:
                probs = F.softmax(next_token_logits.float(), dim=-1)
                top_probs, top_indices = torch.topk(probs, min(top_k, probs.numel()))

                # Cross-attention this token paid to the vision memory, meaned over
                # layers and heads; index 0 is the CLS token, the rest is the 8x8 grid.
                attention = torch.stack([
                    block.cross_attention.last_attention[0, :, pos, :]
                    for block in model.transformer_blocks
                ]).float().mean(dim=(0, 1))                     # (layers, heads, 65) -> (65,)
                patch_attention = (
                    attention[1:].reshape(PATCH_GRID, PATCH_GRID).cpu().numpy().astype(np.float32)
                )

                trace.append({
                    'word': tokenizer.idx_to_word[next_token],
                    'prob': float(probs[next_token]),
                    'top_k': [
                        (tokenizer.idx_to_word[int(idx)], float(p))
                        for p, idx in zip(top_probs.tolist(), top_indices.tolist())
                    ],
                    'attention': patch_attention,
                })

            input_tokens.append(next_token)

            if len(input_tokens) >= MAX_SEQ_LEN:
                break
    finally:
        for block in model.transformer_blocks:
            block.cross_attention.store_attention = False
            block.cross_attention.last_attention = None

    # Decode only the generated answer portion (skip BOS and question)
    response_tokens = input_tokens[len(q_tokens) + 1:]
    response = tokenizer.decode(response_tokens)
    return response, trace


@torch.no_grad()
def generate_response_traced(model, image, question, top_k=3):
    """Greedy-generate a response, plus a per-token introspection trace.

    Returns (response, trace). trace has one dict per generated token -- so its
    entries line up 1:1 with the words of the returned response -- each with:
      'word'      the chosen token's word
      'prob'      that token's softmax probability
      'top_k'     [(word, prob), ...] for the top_k tokens, descending
      'attention' (8, 8) float32 cross-attention over the image patches,
                  averaged across layers and heads (CLS column dropped)
    """
    return _generate(model, image, question, top_k=top_k, with_trace=True)


@torch.no_grad()
def generate_response(model, image, question):
    """Greedy-generate a response for an image and question.

    No introspection trace -- see generate_response_traced for that.
    """
    return _generate(model, image, question)[0]