# Toy Vision-Language Model (VLM) with Chain-of-Thought Reasoning

A PyTorch implementation demonstrating multimodal AI with interpretable reasoning capabilities.

## Project Overview

This toy Vision-Language Model (VLM) can understand multi-shape scenes and answer complex questions using **chain-of-thought reasoning**. The model generates step-by-step rationales before providing final answers, enabling compositional reasoning over counting, comparison, and spatial relations.

## Project Structure

- **model.py**: Neural architectures with spatial vision encoder, auxiliary heads, and CoT generation
- **text.py**: Tokenizer with special reasoning tokens (`<REASON>`, `<SEP>`, `<FINAL>`)
- **shapes.py**: Multi-shape RGB image generation with metadata (colors, sizes, positions)
- **questions.py**: Template system + RationaleGenerator for program traces
- **train_model.py**: Curriculum learning with weighted loss components
- **test_model.py**: Interactive GUI for model inference
- **evaluate.py**: Evaluation suite with difficulty-based testing

## Key Features

### 🧠 Chain-of-Thought Reasoning
The model explains its reasoning before answering:
```
Question: "are there more circles than squares?"
Rationale: "count circles . found 2 . count squares . found 3 . compare 2 vs 3 . 2 is less ."
Answer: "no"
```

### 🎨 Multi-Shape Scenes
- **RGB images** with 2-4 shapes per scene
- **3 colors**: red, green, blue
- **3 sizes**: small, medium, large
- **5 shape types**: square, circle, rectangle, cross, triangle
- **Metadata tracking**: shape type, color, size, position

### 📊 Spatial Vision Encoding
- **8×8 spatial tokens** (64 tokens) instead of global pooling
- **Learnable 2D positional embeddings**
- Preserves spatial information for counting and localization

### 🎯 Auxiliary Heads
- **Per-token**: shape classifier (6 classes), size classifier (3 classes)
- **Global**: count predictors for each shape type (0-4)
- Multi-task supervision helps disentangle visual features

### 📚 Curriculum Learning
**Epochs 0-2** (Easy): Existence, identification
- Loss weights: rationale=2.0, answer=1.0, aux=0.5

**Epochs 3-5** (Medium): Counting, color/size queries
- Loss weights: rationale=1.0, answer=1.5, aux=0.3

**Epochs 6-9** (Hard): Comparison, multi-hop reasoning
- Loss weights: rationale=0.5, answer=2.0, aux=0.2

## Architecture Components

### ToyVLM (Enhanced)
- **SimpleVisionEncoder**: CNN → 8×8 spatial feature map → 64 tokens with positional embeddings
- **Transformer decoder**: 6 layers, 4 heads (increased from 4 layers)
- **Cross-attention**: Text attends to all 64 vision tokens
- **AuxiliaryHeads**: Shape/size classifiers + count predictors

### SimpleTokenizer (Extended)
- **Vocabulary**: ~80 tokens including reasoning words
- **Special tokens**: `<Q>`, `<REASON>`, `<SEP>`, `<FINAL>`
- **Max sequence length**: 40 tokens (increased from 20)
- **Format**: `<START> question <Q> <REASON> rationale <SEP> <FINAL> answer <END>`

### RationaleGenerator
Generates structured program traces for:
- **Easy**: Existence, identification
- **Medium**: Counting, color/size queries
- **Hard**: Comparison, multi-hop reasoning

## Model Configuration

Current hyperparameters:
- Image size: 64×64 pixels (RGB)
- Hidden dimension: 256
- Transformer: 6 layers, 4 attention heads
- Vision tokens: 64 (8×8 grid)
- Max sequence length: 40 tokens
- Batch size: 32
- Training epochs: 10
- Learning rate: 2e-4 with StepLR decay

## Installation

### Recommended: use a project-local virtual environment (`.venv`)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Always activate the venv before running any scripts: `source .venv/bin/activate`.
- On Apple Silicon, PyTorch will run on `mps` automatically; on NVIDIA GPUs, install a CUDA-enabled PyTorch per the official guide if needed (see the PyTorch install selector under "Get Started").

Alternative (not recommended): install globally
```bash
pip install -r requirements.txt
```

## Running the Project

### Training (with Curriculum Learning)
```bash
source .venv/bin/activate
python train_model.py
```
- Trains for 10 epochs with curriculum learning (easy → medium → hard)
- Saves model as `toy_vlm_cot.pth` and vocabulary as `tokenizer_vocab.json`
- Displays per-epoch breakdown of rationale, answer, and auxiliary losses

### Evaluation
```bash
source .venv/bin/activate
python evaluate.py
```
- Evaluates on 100 samples per difficulty level
- Reports exact-match accuracy
- Shows example predictions with rationales

### Interactive GUI
```bash
source .venv/bin/activate
python test_model.py
```
Launches a Tkinter GUI for visual interaction with the trained model.

### Device/runtime notes
- Training auto-detects device via `runtime.py` (CUDA, MPS, or CPU) and configures AMP/autocast and dataloader settings accordingly. No extra flags are required for typical Mac (MPS) or NVIDIA GPU setups.

#### GUI Features
- **Question History**: Navigate previous questions using ↑/↓ arrow keys
- **Auto-focus**: Question input box has focus by default for immediate typing
- **Real-time Interaction**: Ask questions about generated shapes and get instant responses
- **CoT Display**: See both rationale and final answer

## Dependencies

See `requirements.txt` for the complete list of dependencies:
- **torch**: PyTorch with MPS support for Apple Silicon
- **numpy**: Numerical computing
- **tqdm**: Progress bars during training
- **jinja2**: Question template rendering
- **pillow**: Image processing and rotation
- **tkinter**: GUI framework (usually included with Python)

## Example Questions

### Easy (Existence/Identification)
- "is there a circle?"
- "what shapes do you see?"

### Medium (Counting/Attributes)
- "how many circles are there?"
- "how many red shapes are there?"
- "are there any large shapes?"

### Hard (Comparison/Multi-hop)
- "are there more circles than squares?"
- "are there more red shapes than blue shapes?"

## Implementation Details

### Loss Function
```python
total_loss = (weight_rationale * rationale_loss +
              weight_answer * answer_loss +
              weight_aux * aux_loss)
```
- **Rationale loss**: CE on tokens between `<REASON>` and `<SEP>`
- **Answer loss**: CE on tokens between `<FINAL>` and `<END>`
- **Auxiliary loss**: CE on count predictions for each shape type

### Generation Process
1. Encode question: `<START> question <Q> <REASON>`
2. Generate rationale tokens until `<SEP>`
3. Generate `<FINAL>` marker
4. Generate answer tokens until `<END>`
5. Parse and return (rationale, answer)

## Future Enhancements
- Spatial relation questions (left/right/above/below)
- Per-token shape/size supervision with spatial ground truth
- Attention visualization for reasoning steps
- Temperature/top-k sampling strategies
- Failure mode analysis
