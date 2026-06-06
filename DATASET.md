# Dataset

This repository includes 360 sample image/mask pairs for running the training pipeline and demonstrating the expected data layout:

- `picture/source360/`: source portrait images
- `picture/mask360/`: corresponding binary portrait masks

Each source image and mask should share the same file stem. For example:

```text
picture/source360/003ff269-1cec-4ec5-8782-a9566afbe05f.png
picture/mask360/003ff269-1cec-4ec5-8782-a9566afbe05f.png
```

## Format

- Source images are loaded as RGB images.
- Masks are loaded as grayscale images.
- Masks are binarized during training.
- Supported suffixes are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

## Using Custom Data

To train on your own data, place images and masks in separate directories and keep matching file stems:

```text
my_data/images/person_001.jpg
my_data/masks/person_001.png
```

Then run:

```bash
python3 train.py --image-dir my_data/images --mask-dir my_data/masks
```

## License Scope

The MIT License in `LICENSE` applies to the source code. Image datasets and annotations can have separate licensing requirements, so verify the rights for any data you add or redistribute.
