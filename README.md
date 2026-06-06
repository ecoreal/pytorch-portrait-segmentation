# Portrait Segmentation with PyTorch U-Net

A lightweight PyTorch implementation of U-Net for binary portrait segmentation.
The model predicts a foreground mask for portrait images and can be used as a simple baseline for human/portrait segmentation experiments.

## Highlights

- U-Net encoder-decoder architecture for binary segmentation.
- 360 sample image/mask pairs for quickly running the training pipeline.
- Configurable image directory, mask directory, epochs, batch size, learning rate, image size, and checkpoint path.
- Single-image inference script that saves a binary mask.

## Preview

| Source | Mask |
| --- | --- |
| ![source](picture/source360/003ff269-1cec-4ec5-8782-a9566afbe05f.png) | ![mask](picture/mask360/003ff269-1cec-4ec5-8782-a9566afbe05f.png) |

## Project Structure

```text
.
├── dataset.py          # SegmentationDataset: image/mask pair loading and preprocessing
├── model.py            # U-Net model definition
├── train.py            # Training entrypoint
├── predict.py          # Single-image mask prediction entrypoint
├── picture/
│   ├── source360/      # 360 sample input images
│   └── mask360/        # 360 sample binary masks
├── DATASET.md          # Dataset layout and usage notes
├── requirements.txt    # Python dependencies
└── LICENSE             # MIT license for source code
```

## Installation

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

If you need a CUDA-specific PyTorch build, install `torch` and `torchvision` following the official PyTorch command for your CUDA version, then install the remaining dependencies.

## Dataset Layout

The training code expects source images and masks to share the same file stem:

```text
picture/source360/example.png
picture/mask360/example.png
```

Supported image suffixes are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.
Masks are loaded as grayscale images and converted to binary tensors during training.

## Train

Train with the bundled sample dataset:

```bash
python3 train.py --epochs 10 --batch-size 4 --output model.pth
```

Use a custom dataset:

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

## Predict

After training, generate a binary mask for a single image:

```bash
python3 predict.py picture/source360/003ff269-1cec-4ec5-8782-a9566afbe05f.png model.pth --output output_mask.png
```

Optional flags:

- `--image-size`: override the inference resize size.
- `--threshold`: foreground threshold, default `0.5`.

## Limitations

- This repository is a compact baseline rather than a production segmentation system.
- The bundled sample dataset is small, so model quality depends heavily on additional data, annotation quality, augmentation, and training schedule.
- No pretrained checkpoint is included. Train a model first before running inference.

## Roadmap

- Add train/validation split and metrics such as Dice score and IoU.
- Add data augmentation for stronger generalization.
- Add batch inference and alpha-matte foreground extraction.
- Export to ONNX for deployment.

## License

Source code is released under the MIT License. See [LICENSE](LICENSE).

Dataset release and reuse notes are documented separately in [DATASET.md](DATASET.md).
