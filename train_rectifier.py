import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.sr_rectify_dataset import SRRectifyDataset
from models.sr_modules import BasicSRProcessor
from models.velocity import RectifierUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train SR rectifier: SR(D(x)) -> x")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with real training images")
    parser.add_argument("--save_path", type=str, default="rectifier.pth")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--sr_model_name", type=str, default="RealESRGAN_x4plus")
    parser.add_argument("--sr_tile", type=int, default=512)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def freeze_sr(sr_processor: BasicSRProcessor):
    if hasattr(sr_processor, "upsampler") and hasattr(sr_processor.upsampler, "model"):
        for p in sr_processor.upsampler.model.parameters():
            p.requires_grad = False
        sr_processor.upsampler.model.eval()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    dataset = SRRectifyDataset(args.img_dir, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    sr = BasicSRProcessor(
        scale=args.sr_scale,
        model_name=args.sr_model_name,
        device=device,
        tile=args.sr_tile,
    )
    freeze_sr(sr)

    rectifier = RectifierUNet(c_in=3).to(device)
    optimizer = torch.optim.Adam(rectifier.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        rectifier.train()
        total_loss = 0.0

        for x in loader:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                x_sr = sr.sr_process(x)

            x_hat = rectifier(x_sr)
            loss = F.l1_loss(x_hat, x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"[Epoch {epoch + 1}/{args.epochs}] L1: {avg_loss:.6f}")

        if (epoch + 1) % args.save_every == 0:
            torch.save(rectifier.state_dict(), args.save_path)

    torch.save(rectifier.state_dict(), args.save_path)
    print(f"Training finished. Saved to: {args.save_path}")


if __name__ == "__main__":
    main()
