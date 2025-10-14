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

## Key Architecture Components

- **ToyVLM**: Main vision-language model class
  - `SimpleViTEncoder`: ViT-style patch encoder (patch size 8) with CLS token and learned positional embeddings; outputs CLS + 8×8 patch tokens aligned to `hidden_dim`
  - Transformer decoder with multi-head attention (4 layers, 8 heads) using cross-attention from text to vision memory tokens
  - Shared embedding dimension across image and text; both use learned positional embeddings
  
- **SimpleTokenizer**: Custom word-based tokenizer for shape domain
  - Vocabulary: 50 tokens (minimal set for shapes, questions, and answers derived from question set)  
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
- Samples: 500 × BATCH_SIZE (default 32k)
- Training epochs: 10
- Learning rate: 4e-4
- Optimizer: AdamW (weight_decay 0.01)
- Scheduler: LambdaLR with linear warmup (WARMUP_STEPS=500) and cosine decay over TOTAL_STEPS

## Installation

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Running the Project

### Training
```bash
python train_model.py
```
This will train the model and save it as `toy_vlm.pth` along with the vocabulary.

### Interactive GUI
```bash
python test_model.py
```
Launches a Tkinter GUI for visual interaction with the trained model.

#### GUI Features
- **Question History**: Navigate previous questions using ↑/↓ arrow keys
- **Auto-focus**: Question input box has focus by default for immediate typing
- **Real-time Interaction**: Ask questions about generated shapes and get instant responses

## Dependencies

See `requirements.txt` for the complete list of dependencies:
- **torch**: PyTorch with MPS support for Apple Silicon
- **numpy**: Numerical computing
- **tqdm**: Progress bars during training
- **jinja2**: Question template rendering
- **pillow**: Image processing and rotation
- **tkinter**: GUI framework (usually included with Python)

## Known Limitations

1. **Limited question variety**: Only 6 basic templates in questions.txt
2. **Simple vocabulary**: Vocab may need expansion for complex questions
5. **Sequence length**: 20 tokens may be limiting for longer conversations
