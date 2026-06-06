import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dataset
import model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a U-Net model for portrait segmentation.")
    parser.add_argument("--image-dir", default="picture/source360", help="Directory containing training images.")
    parser.add_argument("--mask-dir", default="picture/mask360", help="Directory containing binary mask images.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--image-size", type=int, default=256, help="Square training size in pixels.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--output", default="model.pth", help="Path for the saved checkpoint.")
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model.UNet(n_class=1).to(device)

    train_dataset = dataset.SegmentationDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")

    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = net(images)
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            print(f"\rEpoch {epoch + 1}/{args.epochs}, Batch {i + 1}/{total_batches}, Loss: {avg_loss:.6f}", end="")

        print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "image_size": args.image_size,
            "n_class": 1,
        },
        output_path,
    )
    print(f"Model saved to {output_path}")
    print("Finished Training")


if __name__ == "__main__":
    train(parse_args())
