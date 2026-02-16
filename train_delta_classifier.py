import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.sr_binary_dataset import SRBinaryDataset
from models.residual_rectifier import ResidualRectifierCNN
from models.delta_classifier import SmallCNN


def parse_args():
    p = argparse.ArgumentParser("Train classifier on delta = r - R(r)")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--sr_cache_root", type=str, required=True)
    p.add_argument("--rect_ckpt", type=str, required=True)
    p.add_argument("--save_path", type=str, default="clf_delta.pth")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_abs", action="store_true")
    p.add_argument("--rect_width", type=int, default=64)
    p.add_argument("--rect_depth", type=int, default=10)
    p.add_argument("--clf_width", type=int, default=64)
    return p.parse_args()


@torch.no_grad()
def make_delta(x, x_sr, rectifier, use_abs=False):
    r = x_sr - x
    r_hat = rectifier(r)
    delta = r - r_hat
    if use_abs:
        delta = delta.abs()
    return delta


def load_state_dict_clean(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def main():
    args = parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = True

    ds = SRBinaryDataset(args.data_root, args.sr_cache_root, image_size=args.image_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    rect = ResidualRectifierCNN(c_in=3, width=args.rect_width, depth=args.rect_depth).to(device)
    rect.load_state_dict(load_state_dict_clean(args.rect_ckpt, device))
    rect.eval()
    for p in rect.parameters():
        p.requires_grad_(False)

    clf = SmallCNN(c_in=3, width=args.clf_width).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for ep in range(args.epochs):
        clf.train()
        total = 0.0
        pbar = tqdm(dl, desc=f"[Classifier] epoch {ep + 1}/{args.epochs}")
        for step, (x, x_sr, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            x_sr = x_sr.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().unsqueeze(1)

            with torch.no_grad():
                delta = make_delta(x, x_sr, rect, use_abs=args.use_abs)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                logits = clf(delta)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total / step):.4f}")

        torch.save(clf.state_dict(), args.save_path)
        print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
