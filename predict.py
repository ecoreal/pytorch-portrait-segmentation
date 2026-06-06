import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import model


def parse_args():
    parser = argparse.ArgumentParser(description="Run portrait segmentation with a trained U-Net checkpoint.")
    parser.add_argument("image", help="Input portrait image.")
    parser.add_argument("model", help="Path to a checkpoint produced by train.py.")
    parser.add_argument("--output", default="output_mask.png", help="Output mask path.")
    parser.add_argument("--image-size", type=int, default=None, help="Override inference size in pixels.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Foreground probability threshold.")
    return parser.parse_args()


def load_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("image_size", 256)

    return checkpoint, 256


def predict(image_path, model_path, output_path, image_size=None, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict, checkpoint_image_size = load_checkpoint(model_path, device)
    image_size = image_size or checkpoint_image_size

    net = model.UNet(n_class=1).to(device)
    net.load_state_dict(state_dict)
    net.eval()

    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)
    resized = TF.resize(image, (image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(tensor)
        logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        probability = torch.sigmoid(logits)[0, 0].cpu()
        mask = (probability >= threshold).float()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(mask).save(output_path)
    return output_path


if __name__ == "__main__":
    args = parse_args()
    saved_path = predict(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        image_size=args.image_size,
        threshold=args.threshold,
    )
    print(f"Mask saved to {saved_path}")
