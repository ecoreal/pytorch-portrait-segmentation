from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = _normalize_image_size(image_size)
        self.images = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self._find_mask(image_path.stem)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_size is not None:
            image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).to(torch.float32)
        return image, mask

    def _find_mask(self, stem):
        for suffix in IMAGE_EXTENSIONS:
            mask_path = self.mask_dir / f"{stem}{suffix}"
            if mask_path.exists():
                return mask_path

        raise FileNotFoundError(f"No mask found for image stem '{stem}' in {self.mask_dir}")


def _normalize_image_size(image_size):
    if image_size is None:
        return None

    if isinstance(image_size, int):
        return (image_size, image_size)

    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a pair of integers")

    return tuple(int(value) for value in image_size)
