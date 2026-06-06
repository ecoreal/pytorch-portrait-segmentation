# Portraint Segmentation

一个基于 PyTorch U-Net 的轻量级人像分割项目，用于从人像图片中预测前景人物的二值 mask。

> Note: 仓库名沿用了历史拼写 `Portraint_segmentation`。如果后续希望项目更规范，可以在 GitHub 上改名为 `Portrait_segmentation` 或 `portrait-segmentation`。

## Features

- 使用 U-Net 编码器-解码器结构进行二值分割。
- 内置 360 对样例图片与 mask，便于快速跑通训练流程。
- 支持自定义图片目录、mask 目录、训练轮数、batch size、学习率和模型保存路径。
- 支持加载训练后的 checkpoint，对单张图片输出二值 mask。

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
├── DATASET.md          # Dataset layout and release notes
├── requirements.txt    # Python dependencies
└── LICENSE             # MIT license for source code
```

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- Pillow

Install dependencies:

```bash
pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install `torch` and `torchvision` following the official PyTorch command for your CUDA version, then install the remaining dependencies.

## Dataset Format

The training code expects source images and masks to share the same file stem:

```text
picture/source360/example.png
picture/mask360/example.png
```

Supported image suffixes are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`. Masks are loaded as grayscale images and converted to binary tensors during training.

## Train

Run with the bundled sample dataset:

```bash
python train.py --epochs 10 --batch-size 4 --output model.pth
```

Use a custom dataset:

```bash
python train.py \
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

After training, generate a binary mask for one image:

```bash
python predict.py picture/source360/003ff269-1cec-4ec5-8782-a9566afbe05f.png model.pth --output output_mask.png
```

Optional flags:

- `--image-size`: override the inference resize size.
- `--threshold`: foreground threshold, default `0.5`.

## Notes

- The current implementation is intentionally simple and is suitable as a learning baseline or small-data experiment.
- The bundled dataset is small, so model quality will depend heavily on data diversity, annotation quality, augmentation, and training schedule.
- Generated files such as `model.pth`, prediction outputs, virtual environments, and Python cache files are ignored by `.gitignore`.

## Roadmap

- Add train/validation split and metrics such as Dice score and IoU.
- Add data augmentation for stronger generalization.
- Add batch inference and alpha-matte foreground extraction.
- Export to ONNX for deployment.

## License

Source code is released under the MIT License. See [LICENSE](LICENSE).

Dataset release and reuse notes are documented separately in [DATASET.md](DATASET.md).
