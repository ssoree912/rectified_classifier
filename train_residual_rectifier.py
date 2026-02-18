import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.sr_residual_rectify_dataset import SRResidualRectifyDataset
from models.residual_rectifier import ResidualRectifierCNN


def parse_args():
    p = argparse.ArgumentParser("Train residual rectifier: r_noisy -> r_clean (real-only)")
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--sr_cache_root", type=str, required=True)
    p.add_argument("--save_path", type=str, default="rect_res.pth")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--noise_std", type=float, default=0.02)
    p.add_argument("--use_abs", action="store_true")
    p.add_argument("--allow_mixed", action="store_true", help="Disable real-only safety checks")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def ensure_real_only_dir(img_dir: str, allow_mixed: bool = False):
    if allow_mixed:
        return

    p = Path(img_dir).resolve()
    if not p.exists():
        raise FileNotFoundError(f"img_dir not found: {p}")

    child_dirs = {d.name.lower() for d in p.iterdir() if d.is_dir()}

    # Common dataset layouts that contain both classes under one root.
    if {"real", "fake"}.issubset(child_dirs):
        raise ValueError(
            f"img_dir must be real-only, but found both real/fake under {p}. "
            "Use a real-only subfolder, e.g. .../real"
        )
    if {"nature", "ai"}.issubset(child_dirs):
        raise ValueError(
            f"img_dir must be real-only, but found both nature/ai under {p}. "
            "Use a real-only subfolder, e.g. .../nature"
        )

    # Additional guard for common fake folder names directly under root.
    if "fake" in child_dirs or "ai" in child_dirs:
        raise ValueError(
            f"img_dir appears mixed/non-real-only ({p}). "
            "Set --img_dir to a real-only directory."
        )

    # Guard for generic multi-generator layouts: root/real + many fake folders.
    if "real" in child_dirs and len(child_dirs - {"real"}) > 0:
        raise ValueError(
            f"img_dir appears mixed/non-real-only ({p}). "
            "Found `real` plus additional sibling folders. "
            "Set --img_dir to the real-only directory (e.g. .../real)."
        )
    if "nature" in child_dirs and len(child_dirs - {"nature"}) > 0:
        raise ValueError(
            f"img_dir appears mixed/non-real-only ({p}). "
            "Found `nature` plus additional sibling folders. "
            "Set --img_dir to the real-only directory (e.g. .../nature)."
        )


def main():
    args = parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = True
    ensure_real_only_dir(args.img_dir, allow_mixed=args.allow_mixed)

    ds = SRResidualRectifyDataset(
        args.img_dir,
        args.sr_cache_root,
        image_size=args.image_size,
        noise_std=args.noise_std,
        use_abs=args.use_abs,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = ResidualRectifierCNN(c_in=3, width=64, depth=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        pbar = tqdm(dl, desc=f"[Rectifier] epoch {ep + 1}/{args.epochs}")
        for step, (r_clean, r_noisy) in enumerate(pbar, start=1):
            r_clean = r_clean.to(device, non_blocking=True)
            r_noisy = r_noisy.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                r_hat = model(r_noisy)
                loss = F.l1_loss(r_hat, r_clean)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{(total / step):.5f}")

        torch.save(model.state_dict(), args.save_path)
        print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
