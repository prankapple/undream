
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# ==========================================
# UN-DREAM
# PyTorch DeepDream reconstruction
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# DATASET
# ==========================================

class DreamDataset(Dataset):
    def __init__(self, image, count=500):
        self.image = image
        self.count = count
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        original = self.to_tensor(self.image)
        dreamed = original.clone()

        # Noise
        if random.random() < 0.8:
            dreamed += torch.randn_like(dreamed) * random.uniform(0.02, 0.2)

        # Brightness
        if random.random() < 0.7:
            dreamed *= random.uniform(0.6, 1.5)

        # Color distortion
        if random.random() < 0.7:
            colors = torch.tensor([
                random.uniform(0.7, 1.3),
                random.uniform(0.7, 1.3),
                random.uniform(0.7, 1.3)
            ]).view(3, 1, 1)

            dreamed *= colors

        # Blur
        if random.random() < 0.5:
            dreamed = F.avg_pool2d(
                dreamed.unsqueeze(0),
                3, stride=1, padding=1
            ).squeeze(0)

        return dreamed.clamp(0, 1), original


# ==========================================
# NEURAL NETWORK
# ==========================================

class Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU()
        )

        self.skip = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class UNDream(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = Block(3, 32)
        self.enc2 = Block(32, 64)
        self.enc3 = Block(64, 128)

        self.middle = Block(128, 128)

        self.dec3 = Block(256, 64)
        self.dec2 = Block(128, 32)
        self.dec1 = Block(64, 32)

        self.output = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))

        m = self.middle(F.avg_pool2d(c, 2))

        x = F.interpolate(
            m, size=c.shape[-2:],
            mode="bilinear", align_corners=False
        )
        x = self.dec3(torch.cat([x, c], dim=1))

        x = F.interpolate(
            x, size=b.shape[-2:],
            mode="bilinear", align_corners=False
        )
        x = self.dec2(torch.cat([x, b], dim=1))

        x = F.interpolate(
            x, size=a.shape[-2:],
            mode="bilinear", align_corners=False
        )
        x = self.dec1(torch.cat([x, a], dim=1))

        return torch.sigmoid(self.output(x))


# ==========================================
# MODEL SAVE / LOAD
# ==========================================

def save_model(model, optimizer, epoch, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "architecture": "UNDream"
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    if checkpoint.get("architecture") != "UNDream":
        raise ValueError("Invalid model checkpoint")

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"]


# ==========================================
# INFERENCE
# ==========================================

class UNDreamEngine:
    def __init__(self, checkpoint="undream.pth"):
        self.device = DEVICE
        self.model = UNDream().to(self.device)

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"Model not found: {checkpoint}. Train one first."
            )

        data = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=True
        )

        if data.get("architecture") != "UNDream":
            raise ValueError("Invalid UN-DREAM checkpoint")

        self.model.load_state_dict(data["model"])
        self.model.eval()

        self.epoch = data["epoch"]

        print("UN-DREAM model loaded!")
        print("Epochs:", self.epoch)
        print("Device:", self.device)

    @torch.no_grad()
    def restore(self, input_path, output_path="restored.png",
                size=256):

        image = Image.open(input_path).convert("RGB")

        original_size = image.size

        image = image.resize((size, size))

        tensor = transforms.ToTensor()(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        result = self.model(tensor)

        result = result.squeeze(0).cpu()

        output = transforms.ToPILImage()(result)

        output = output.resize(original_size)
        output.save(output_path)

        print("Reconstruction saved:", output_path)

        return output


# ==========================================
# TRAINING
# ==========================================

def train(args):
    image = Image.open(args.input).convert("RGB")
    image = image.resize((args.size, args.size))

    dataset = DreamDataset(image, count=500)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    model = UNDream().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    start_epoch = 0

    if os.path.exists(args.checkpoint) and not args.retrain:
        print("Loading previous checkpoint...")

        start_epoch = load_model(
            model, optimizer, args.checkpoint
        )

        print("Resuming at epoch:", start_epoch)

    else:
        print("Training new model...")

    model.train()

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0

        for dreamed, original in loader:
            dreamed = dreamed.to(DEVICE)
            original = original.to(DEVICE)

            reconstructed = model(dreamed)

            loss = (
                F.l1_loss(reconstructed, original)
                + 0.5 * F.mse_loss(reconstructed, original)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Loss: {average_loss:.6f}"
        )

        save_model(
            model, optimizer, epoch + 1, args.checkpoint
        )

        print("Checkpoint saved:", args.checkpoint)

    print("Training complete!")


# ==========================================
# COMMAND LINE
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="UN-DREAM neural image reconstruction"
    )

    parser.add_argument(
        "--input", required=True,
        help="Input image"
    )

    parser.add_argument(
        "--output", default="undreamed.png"
    )

    parser.add_argument(
        "--checkpoint", default="undream.pth"
    )

    parser.add_argument(
        "--epochs", type=int, default=100
    )

    parser.add_argument(
        "--size", type=int, default=256
    )

    parser.add_argument(
        "--retrain", action="store_true"
    )

    parser.add_argument(
        "--use-model", action="store_true",
        help="Load existing model instead of training"
    )

    args = parser.parse_args()

    print("=" * 45)
    print("             UN-DREAM")
    print("=" * 45)

    print("Device:", DEVICE)

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if args.use_model:
        engine = UNDreamEngine(args.checkpoint)

        engine.restore(
            args.input,
            args.output,
            args.size
        )

    else:
        train(args)

        # Automatically reconstruct using trained model
        engine = UNDreamEngine(args.checkpoint)

        engine.restore(
            args.input,
            args.output,
            args.size
        )

    print("\nUN-DREAM COMPLETE")


if __name__ == "__main__":
    main()
