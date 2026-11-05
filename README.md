# UN-DREAM

**An experimental PyTorch-based neural network for DeepDream image reconstruction.**

UN-DREAM is an open-source image reconstruction experiment that uses a custom convolutional neural network built entirely with PyTorch. It aims to explore reversing visual distortions and reconstructing images without relying on pretrained models.

## Features

- 🧠 Custom CNN architecture built with PyTorch
- 🚫 No pretrained models required
- ⚡ Automatic CUDA GPU acceleration
- 💾 Automatic model checkpoint saving
- ♻️ Resume interrupted training
- 🖼️ Image reconstruction and inference
- 🔧 Configurable training epochs and image resolution
- 📦 Reusable Python inference interface

## Installation

Install the required dependencies:

```bash
pip install torch torchvision pillow
```

Clone the repository:

```bash
git clone https://github.com/prankapple/undream.git
cd undream
```

## Usage

### 1. Train a model

```bash
python undream.py --input deepdream.png
```

This trains a new model and saves its checkpoint as `undream.pth`.

### 2. Continue training

```bash
python undream.py --input deepdream.png --epochs 200
```

Training resumes from the existing checkpoint, if available. The epoch argument specifies the total target epoch count.

### 3. Train from scratch

```bash
python undream.py --input deepdream.png --retrain
```

### 4. Use an existing model

```bash
python undream.py --input deepdream.png --use-model
```

### 5. Customize output

```bash
python undream.py --input deepdream.png --output restored.png --size 512
```

## Python API

UN-DREAM can also be imported into other Python projects.

```python
from undream import UNDreamEngine

model = UNDreamEngine("undream.pth")

model.restore(
    "deepdream.png",
    "restored.png",
    size=256
)
```

## Model Architecture

UN-DREAM uses a custom encoder-decoder CNN with skip connections.

| Component | Channels |
|---|---:|
| Encoder 1 | 32 |
| Encoder 2 | 64 |
| Encoder 3 | 128 |
| Middle | 128 |
| Decoder 3 | 64 |
| Decoder 2 | 32 |
| Decoder 1 | 32 |
| Output | 3 |

The network uses:
- Convolutional layers
- Batch normalization
- GELU activation
- Average pooling
- Bilinear upsampling
- Skip connections

## Checkpoints

Models are saved automatically after every training epoch.

A checkpoint contains:

| Data | Description |
|---|---|
| `model` | Neural network weights |
| `optimizer` | Optimizer state |
| `epoch` | Training progress |
| `architecture` | Model architecture identifier |

Default checkpoint:

```text
undream.pth
```

## Project Structure

```text
undream/
│
├── undream.py
└── README.md
```

## Important Limitations

UN-DREAM is experimental.

The current implementation trains using artificially corrupted versions of the supplied image, with the original image as its reconstruction target.

It does not currently learn to reverse arbitrary DeepDream transformations or recover information permanently lost during image processing.

Exact reconstruction generally requires access to the original image or additional information about the transformation.

## Requirements

- Python 3.10+
- PyTorch
- Torchvision
- Pillow
- Optional NVIDIA CUDA-compatible GPU

## License

MIT License (see `LICENSE`).

---

**UN-DREAM — Exploring the reconstruction of dreamed images.**