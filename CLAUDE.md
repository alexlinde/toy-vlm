# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a toy Vision-Language Model (VLM) implementation in PyTorch that demonstrates basic multimodal AI capabilities. The model can understand simple geometric shapes (square, circle, rectangle, cross, triangle) and answer questions about them.

## Project Structure

- **model.py**: Core neural network architectures and training utilities
- **text.py**: Text processing with StandardTokenizer using transformers
- **shapes.py**: Geometric shape generation for synthetic data
- **questions.py**: Jinja2-based question template system
- **vlm-generate.py**: Training script with dataset generation
- **vlm-execute.py**: Interactive GUI for model inference

## Key Architecture Components

- **ToyVLM**: Main vision-language model class
  - `SimpleVisionEncoder`: CNN for 64x64 grayscale image processing
  - Transformer decoder with multi-head attention (4 layers, 4 heads)
  - Vision features as sequence prefix for multimodal fusion
  
- **SimpleTokenizer**: Custom word-based tokenizer for shape domain
  - Vocabulary: 50 tokens (minimal set for shapes, questions, and answers derived from question set)  
  - Alpha-only preprocessing: strips punctuation and normalizes text
  - Special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`
  - Max sequence length: 20 tokens
  
- **ShapeGenerator**: Synthetic dataset creation
  - 5 shape types: square, circle, rectangle, cross, triangle
  - Random positioning, sizing, and noise injection
  
- **QuestionGenerator**: Template-based Q&A generation
  - Uses `questions.txt` with basic question templates
  - Jinja2 templates support shape identification and yes/no questions

## Hardware Configuration

- **Device**: Hardcoded to Apple Silicon MPS (`torch.device('mps')`)
- **Warning**: No automatic fallback to CPU/CUDA if MPS unavailable

## Model Configuration

Current hyperparameters:
- Image size: 64x64 pixels
- Hidden dimension: 256
- Transformer: 4 layers, 4 attention heads  
- Max sequence length: 20 tokens (reduced from 50)
- Batch size: 25
- Training epochs: 5
- Learning rate: 1e-3

## Running the Project

### Training
```bash
python vlm-generate.py
```
This will train the model and save it as `toy_vlm.pth` along with the vocabulary.

### Interactive GUI
```bash
python vlm-execute.py
```
Launches a Tkinter GUI for visual interaction with the trained model.

#### GUI Features
- **Question History**: Navigate previous questions using ↑/↓ arrow keys
- **Auto-focus**: Question input box has focus by default for immediate typing
- **Real-time Interaction**: Ask questions about generated shapes and get instant responses

## Dependencies

**Required packages** (install manually):
- torch (with MPS support)
- numpy
- tqdm (progress bars)
- jinja2 (question templates)
- pillow (image processing for GUI)
- tkinter (GUI - usually included with Python)

## Known Limitations

1. **Limited question variety**: Only 6 basic templates in questions.txt
2. **Simple vocabulary**: Vocab may need expansion for complex questions
3. **No device fallback**: MPS hardcoding prevents running on other systems
5. **Sequence length**: 20 tokens may be limiting for longer conversations
