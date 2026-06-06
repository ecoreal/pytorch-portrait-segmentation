# Portrait Segmentation with PyTorch U-Net

[English](README.md) | [简体中文](README.zh-CN.md)

A lightweight PyTorch implementation of U-Net for binary portrait segmentation. This project provides a compact training and inference pipeline for predicting foreground masks from portrait images, making it suitable as a baseline for portrait segmentation, human segmentation, and related computer vision experiments.

## Highlights

- Implements a U-Net encoder-decoder network for binary image segmentation.
- Includes 360 sample image/mask pairs for quickly validating the training workflow.
- Supports configurable dataset paths, epochs, batch size, learning rate, image size, and checkpoint output.
- Provides a single-image inference script for generating binary portrait masks.
- Keeps the codebase small and easy to inspect for learning, experimentation, and extension.

## Preview

| Source | Mask |
| --- | --- |
| ![source](picture/source360/003ff269-1cec-4ec5-8782-a9566afbe05f.png) | ![mask](picture/mask360/003ff269-1cec-4ec5-8782-a9566afbe05f.png) |

## Repository Structure

```text
.
├── README.md           # English project documentation
├── README.zh-CN.md     # Simplified Chinese project documentation
├── dataset.py          # Dataset loading and preprocessing
├── model.py            # U-Net model definition
├── train.py            # Training entrypoint
├── predict.py          # Single-image inference entrypoint
├── picture/
│   ├── source360/      # Sample input images
│   └── mask360/        # Sample binary masks
├── DATASET.md          # Dataset format and usage notes
├── requirements.txt    # Python dependencies
└── LICENSE             # Source code license
```

## Installation

Requirements:

- Python 3.9+
- PyTorch
- torchvision
- Pillow

Clone the repository and install dependencies:

```bash
git clone https://github.com/ecoreal/pytorch-portrait-segmentation.git
cd pytorch-portrait-segmentation
pip install -r requirements.txt
```

For CUDA-enabled environments, install the appropriate `torch` and `torchvision` builds from the official PyTorch installation instructions, then install the remaining dependencies.

## Dataset Layout

The training pipeline expects source images and masks to be stored in separate directories with matching file stems:

```text
picture/source360/example.png
picture/mask360/example.png
```

Supported image suffixes are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`. Masks are loaded as grayscale images and binarized during training.

For additional dataset details, see [DATASET.md](DATASET.md).

## Training

Train with the bundled sample dataset:

```bash
python3 train.py --epochs 10 --batch-size 4 --output model.pth
```

Train with a custom dataset:

```bash
python3 train.py \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --image-size 256 \
  --output checkpoints/portrait_unet.pth
```

The saved checkpoint contains:

- `model_state_dict`
- `image_size`
- `n_class`

## Inference

After training, generate a binary mask for a single image:

```bash
python3 predict.py picture/source360/003ff269-1cec-4ec5-8782-a9566afbe05f.png model.pth --output output_mask.png
```

Optional arguments:

- `--image-size`: override the inference resize size.
- `--threshold`: foreground probability threshold. The default value is `0.5`.

## Limitations

- This repository is designed as a compact baseline rather than a production-ready segmentation system.
- The bundled sample dataset is small; model quality depends on additional data, annotation quality, data augmentation, and training configuration.
- No pretrained checkpoint is included. Train a model before running inference.

## Roadmap

- Add train/validation split support and segmentation metrics such as Dice score and IoU.
- Add data augmentation for stronger generalization.
- Add batch inference and foreground extraction utilities.
- Add ONNX export for deployment.

## License

The source code is released under the MIT License. See [LICENSE](LICENSE).

Dataset format and reuse notes are documented in [DATASET.md](DATASET.md).
