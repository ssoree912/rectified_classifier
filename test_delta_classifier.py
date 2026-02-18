import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.delta_classifier import SmallCNN
from models.residual_rectifier import ResidualRectifierCNN


def parse_args():
    p = argparse.ArgumentParser("Evaluate delta classifier from residual cache")
    p.add_argument("--residual_cache_root", type=str, required=True)
    p.add_argument("--rect_ckpt", type=str, required=True)
    p.add_argument("--clf_ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--use_abs", action="store_true")
    p.add_argument("--rect_width", type=int, default=64)
    p.add_argument("--rect_depth", type=int, default=10)
    p.add_argument("--clf_width", type=int, default=64)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--result_folder", type=str, default=None)
    p.add_argument("--exp_name", type=str, default="delta_eval")
    return p.parse_args()


def load_state_dict_clean(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


class ResidualCacheBinaryDataset(Dataset):
    def __init__(self, root, max_items=None):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"residual_cache_root not found: {self.root}")

        real_dir = self.root / "real"
        fake_dir = self.root / "fake"
        if not real_dir.is_dir():
            real_dir = self.root / "nature"
        if not fake_dir.is_dir():
            fake_dir = self.root / "ai"

        if not real_dir.is_dir() or not fake_dir.is_dir():
            raise ValueError(
                f"Could not find class folders under {self.root}. "
                "Expected (real,fake) or (nature,ai)."
            )

        self.items = []
        for d, y in [(real_dir, 0), (fake_dir, 1)]:
            for p in sorted(d.rglob("*.pt")):
                self.items.append((p, y))

        if max_items is not None:
            self.items = self.items[:max_items]
        if not self.items:
            raise ValueError(f"No .pt files found under: {self.root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        r = torch.load(p, map_location="cpu")
        if not torch.is_tensor(r):
            raise TypeError(f"Expected tensor in {p}, got {type(r)}")
        return r.float(), torch.tensor(y, dtype=torch.float32)


def find_best_threshold(y_true, y_pred):
    best_acc = -1.0
    best_th = 0.5
    for th in np.unique(y_pred):
        acc = accuracy_score(y_true, y_pred > th)
        if acc >= best_acc:
            best_acc = acc
            best_th = float(th)
    return best_th


def calculate_acc(y_true, y_pred, thres):
    r_mask = y_true == 0
    f_mask = y_true == 1
    r_acc = accuracy_score(y_true[r_mask], y_pred[r_mask] > thres) if r_mask.any() else float("nan")
    f_acc = accuracy_score(y_true[f_mask], y_pred[f_mask] > thres) if f_mask.any() else float("nan")
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


@torch.no_grad()
def evaluate(rect, clf, loader, device, threshold=0.5, use_abs=False):
    rect.eval()
    clf.eval()
    y_true = []
    y_pred = []
    pbar = tqdm(loader, desc="[Test]")
    for r, y in pbar:
        r = r.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        r_hat = rect(r)
        delta = r - r_hat
        if use_abs:
            delta = delta.abs()
        logits = clf(delta).squeeze(1)
        y_pred.extend(logits.sigmoid().detach().cpu().tolist())
        y_true.extend(y.detach().cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    try:
        ap = average_precision_score(y_true, y_pred)
    except ValueError:
        ap = float("nan")
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, threshold)
    best_th = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_th)
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_th, y_pred, y_true


def main():
    args = parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = True

    ds = ResidualCacheBinaryDataset(args.residual_cache_root, max_items=args.max_items)
    dl_kwargs = dict(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch_factor
    dl = DataLoader(**dl_kwargs)

    rect = ResidualRectifierCNN(c_in=3, width=args.rect_width, depth=args.rect_depth).to(device)
    rect.load_state_dict(load_state_dict_clean(args.rect_ckpt, device), strict=True)

    clf = SmallCNN(c_in=3, width=args.clf_width).to(device)
    clf.load_state_dict(load_state_dict_clean(args.clf_ckpt, device), strict=True)

    ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_th, y_pred, y_true = evaluate(
        rect=rect,
        clf=clf,
        loader=dl,
        device=device,
        threshold=args.threshold,
        use_abs=args.use_abs,
    )
    print(f"AP: {ap:.6f}")
    print(f"ACC@{args.threshold:.3f}: real={r_acc0:.6f} fake={f_acc0:.6f} total={acc0:.6f}")
    print(f"ACC@best_th({best_th:.6f}): real={r_acc1:.6f} fake={f_acc1:.6f} total={acc1:.6f}")

    if args.result_folder:
        result_dir = Path(args.result_folder)
        result_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = Path(args.clf_ckpt).stem
        prefix = f"{args.exp_name}_{ckpt_name}"

        metrics = {
            "residual_cache_root": str(Path(args.residual_cache_root).resolve()),
            "rect_ckpt": str(Path(args.rect_ckpt).resolve()),
            "clf_ckpt": str(Path(args.clf_ckpt).resolve()),
            "num_samples": int(len(ds)),
            "threshold": float(args.threshold),
            "use_abs": bool(args.use_abs),
            "ap": float(ap),
            "acc0_real": float(r_acc0),
            "acc0_fake": float(f_acc0),
            "acc0_total": float(acc0),
            "best_threshold": float(best_th),
            "acc1_real": float(r_acc1),
            "acc1_fake": float(f_acc1),
            "acc1_total": float(acc1),
        }

        with open(result_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=True, indent=2)

        with open(result_dir / f"{prefix}_metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"AP: {ap:.6f}\n")
            f.write(
                f"ACC@{args.threshold:.3f}: real={r_acc0:.6f} fake={f_acc0:.6f} total={acc0:.6f}\n"
            )
            f.write(
                f"ACC@best_th({best_th:.6f}): real={r_acc1:.6f} fake={f_acc1:.6f} total={acc1:.6f}\n"
            )

        np.savez(
            result_dir / f"{prefix}_predictions.npz",
            y_pred=np.asarray(y_pred, dtype=np.float32),
            y_true=np.asarray(y_true, dtype=np.float32),
        )

        # Unified output format with validate.py
        df = pd.DataFrame(
            [
                dict(
                    dataset=Path(args.residual_cache_root).name,
                    ap=float(ap),
                    ap_pct=float(ap * 100.0),
                    r_acc0=float(r_acc0),
                    r_acc0_pct=float(r_acc0 * 100.0),
                    f_acc0=float(f_acc0),
                    f_acc0_pct=float(f_acc0 * 100.0),
                    acc0=float(acc0),
                    acc0_pct=float(acc0 * 100.0),
                    r_acc1=float(r_acc1),
                    r_acc1_pct=float(r_acc1 * 100.0),
                    f_acc1=float(f_acc1),
                    f_acc1_pct=float(f_acc1 * 100.0),
                    acc1=float(acc1),
                    acc1_pct=float(acc1 * 100.0),
                    best_thres=float(best_th),
                ),
                dict(
                    dataset="average",
                    ap=float(ap),
                    ap_pct=float(ap * 100.0),
                    r_acc0=float(r_acc0),
                    r_acc0_pct=float(r_acc0 * 100.0),
                    f_acc0=float(f_acc0),
                    f_acc0_pct=float(f_acc0 * 100.0),
                    acc0=float(acc0),
                    acc0_pct=float(acc0 * 100.0),
                    r_acc1=float(r_acc1),
                    r_acc1_pct=float(r_acc1 * 100.0),
                    f_acc1=float(f_acc1),
                    f_acc1_pct=float(f_acc1 * 100.0),
                    acc1=float(acc1),
                    acc1_pct=float(acc1 * 100.0),
                    best_thres=float(best_th),
                ),
            ]
        )
        df.to_excel(result_dir / "validation.xlsx", index=False)
        np.savez(result_dir / "validation_ypred.npz", np.asarray(y_pred, dtype=np.float32))
        np.savez(result_dir / "validation_ytrue.npz", np.asarray(y_true, dtype=np.float32))
        print(f"Saved results to: {result_dir}")


if __name__ == "__main__":
    main()
