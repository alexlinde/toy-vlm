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
  - Vocabulary: 29 tokens, built deterministically by enumerating every question/answer template in `questions.txt` for every shape
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
- Samples: 1000 × BATCH_SIZE (default 64k)
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
This will train the model and save it as `toy_vlm.pth` along with the vocabulary.

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

1. **Limited question variety**: Only 12 basic templates in questions.txt
2. **Simple vocabulary**: Vocab may need expansion for complex questions
5. **Sequence length**: 20 tokens may be limiting for longer conversations
