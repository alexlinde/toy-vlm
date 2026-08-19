# Toy Vision-Language Model (VLM)

A simple PyTorch implementation demonstrating basic multimodal AI capabilities.

## Project Overview

This is a toy Vision-Language Model (VLM) implementation in PyTorch that demonstrates basic multimodal AI capabilities. The model can understand simple geometric shapes (square, circle, rectangle, cross, triangle) and answer questions about them.

## Project Structure

- **model.py**: Core neural network architectures and training utilities
- **text.py**: Text processing with SimpleTokenizer
- **shapes.py**: Geometric shape generation for synthetic data
- **questions.py**: Jinja2-based question template system
- **train_model.py**: Training script with dataset generation
- **test_model.py**: Interactive GUI for model inference
- **evaluate.py**: Quantitative exact-match evaluation across all question templates

## Key Architecture Components

- **ToyVLM**: Main vision-language model class
  - `SimpleViTEncoder`: ViT-style patch encoder (patch size 8) with CLS token and learned positional embeddings; outputs CLS + 8×8 patch tokens aligned to `hidden_dim`
  - Transformer decoder with multi-head attention (4 layers, 8 heads) using cross-attention from text to vision memory tokens
  - Shared embedding dimension across image and text; both use learned positional embeddings
  
- **SimpleTokenizer**: Custom word-based tokenizer for shape domain
  - Vocabulary: 42 tokens, built deterministically by enumerating every question/answer template in `questions.txt` for every shape
  - Alpha-only preprocessing: strips punctuation and normalizes text
  - Special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`
  - Max sequence length: 20 tokens
  
- **ShapeGenerator**: Synthetic dataset creation
  - 5 shape types: square, circle, rectangle, cross, triangle
  - Random positioning, sizing, rotation, and noise injection
  
- **QuestionGenerator**: Template-based Q&A generation
  - Uses `questions.txt` with basic question templates
  - Jinja2 templates support shape identification and yes/no questions

## Model Configuration

Current hyperparameters:
- Image size: 64x64 pixels
- Hidden dimension: 256
- Transformer: 4 layers, 8 attention heads  
- Max sequence length: 20 tokens
- Batch size: 64
- Samples: 64k by default (total across all ranks, independent of batch size)
- Training epochs: 10
- Learning rate: 4e-4
- Optimizer: AdamW (weight_decay 0.01)
- Scheduler: LambdaLR with linear warmup (defaults to 1% of total steps) and cosine decay over TOTAL_STEPS

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:
```bash
uv sync
```

## Running the Project

### Training
```bash
uv run python train_model.py
```
This will train the model and save it as `toy_vlm.pth` with the vocabulary
bundled inside the checkpoint, so the weights and vocab can never mismatch.

### Single-node multi-GPU training with torchrun

- All local GPUs:
```bash
uv run torchrun --standalone --nproc_per_node=$(uv run python -c "import torch;print(torch.cuda.device_count())") \
  train_model.py --distributed --backend nccl --batch-size 8 --workers 8
```

- Specific number of GPUs (e.g., 4):
```bash
uv run torchrun --standalone --nproc_per_node=4 \
  train_model.py --distributed --backend nccl --batch-size 8 --workers 8
```

- Single GPU (or run without DDP):
```bash
uv run torchrun --standalone --nproc_per_node=1 \
  train_model.py --distributed --backend nccl
# or
uv run python train_model.py
```

Notes:
- Batch size is per process. Global batch = batch_size × nproc_per_node.
- Use backend `nccl` on NVIDIA GPUs, `gloo` for CPU-only.

### Interactive GUI
```bash
uv run python test_model.py
```
Launches a Tkinter GUI for visual interaction with the trained model.

#### GUI Features
- **Question History**: Navigate previous questions using ↑/↓ arrow keys
- **Auto-focus**: Question input box has focus by default for immediate typing
- **Real-time Interaction**: Ask questions about generated shapes and get instant responses

#### Model Introspection
Each answer word is background-coloured by the model's confidence in that token, with
the runner-up alternatives listed underneath. Click a word to paint its cross-attention
over the image as a red heatmap (averaged across layers and heads; toggle with **Show
attention**) — the answer's average map is shown by default. Under the drawing tools, a
linear probe on the frozen vision patch embeddings reports live shape probabilities as
bars, updating as you draw. The probe is trained at the end of `train_model.py` and bundled
into the checkpoint; older checkpoints without one simply show a note instead of the bars.
It reads the patch tokens rather than the CLS token because `SimpleViTEncoder` has no
self-attention, so its CLS output is a learned constant that carries no image information.

### Evaluation
```bash
uv run python evaluate.py
```
Measures exact-match accuracy of greedy generation against `--checkpoint` (default
`toy_vlm.pth`), sampling `--samples` random shapes per question template (default 200)
from `questions.txt`. Reports per-template accuracy plus rollups for three families -
**identification** (open "what shape is this"-style questions), **yes** and **no**
(positive/negative verification questions) - each against its own majority-class
baseline, along with overall accuracy and counts of empty generations/exceptions.

By default images are clean (noise-free), matching the interactive GUI; pass `--noise`
to evaluate under the noisier conditions used during training. Useful flags:
```bash
uv run python evaluate.py --checkpoint toy_vlm.pth --samples 200 --seed 0 --noise
```

## Dependencies

See `pyproject.toml` for the complete list of dependencies:
- **torch**: PyTorch with MPS support for Apple Silicon
- **numpy**: Numerical computing
- **tqdm**: Progress bars during training
- **jinja2**: Question template rendering
- **pillow**: Image processing and rotation
- **tkinter**: GUI framework (usually included with Python)

## Known Limitations

1. **Limited question variety**: Only 24 basic templates in questions.txt
2. **Simple vocabulary**: Vocab may need expansion for complex questions
3. **Sequence length**: 20 tokens may be limiting for longer conversations

## Deliberate Simplicity Choices

A few places where this code intentionally differs from modern transformer
practice. Each alternative is load-bearing at some scale of depth, vocabulary,
or precision — and this project sits comfortably below all of those
thresholds, so the simpler (or more default) version is kept for readability:

- **Post-norm blocks** (`norm(x + sublayer(x))`, the original 2017 layout).
  Modern stacks use pre-norm because post-norm destabilizes training past
  ~10-12 layers without careful warmup. At 4 layers the gradient path is
  short enough that it trains without issue.
- **No weight tying** between the token embedding and the output projection.
  Tying saves parameters when the vocabulary is large relative to the model;
  with a 42-token vocab the untied head costs ~0.4% of total parameters,
  and separate matrices are easier to reason about.
- **Default N(0,1) embedding init** instead of GPT-style N(0, 0.02). The
  small init matters mainly when the embedding is *tied* to the output head
  (unit-variance weights there produce huge initial logits); untied, with a
  LayerNorm downstream of the embedding sum, the default is harmless.

If you copy this code into a deeper, larger-vocab, or tied-head model, revisit
all three.
