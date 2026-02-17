import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.residual_binary_cache_dataset import ResidualBinaryCacheDataset
from dataset.sr_binary_dataset import SRBinaryDataset
from models.delta_classifier import SmallCNN
from models.residual_rectifier import ResidualRectifierCNN


def parse_args():
    p = argparse.ArgumentParser("Evaluate classifier on delta = r - R(r)")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--sr_cache_root", type=str, default=None)
    p.add_argument("--residual_cache_root", type=str, default=None)
    p.add_argument("--rect_ckpt", type=str, required=True)
    p.add_argument("--clf_ckpt", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_abs", action="store_true")
    p.add_argument("--rect_width", type=int, default=64)
    p.add_argument("--rect_depth", type=int, default=10)
    p.add_argument("--clf_width", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save_npz", type=str, default=None)
    return p.parse_args()


def load_state_dict_clean(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


@torch.no_grad()
def make_delta(x, x_sr, rectifier, use_abs=False):
    r = x_sr - x
    r_hat = rectifier(r)
    delta = r - r_hat
    if use_abs:
        delta = delta.abs()
    return delta


@torch.no_grad()
def make_delta_from_residual(r, rectifier, use_abs=False):
    r_hat = rectifier(r)
    delta = r - r_hat
    if use_abs:
        delta = delta.abs()
    return delta


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
def evaluate(clf, rect, loader, device, use_abs=False, threshold=0.5, use_residual_cache=False):
    clf.eval()
    rect.eval()
    y_true = []
    y_pred = []
    pbar = tqdm(loader, desc="[Test]")
    for batch in pbar:
        if use_residual_cache:
            r, y = batch
            r = r.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            delta = make_delta_from_residual(r, rect, use_abs=use_abs)
        else:
            x, x_sr, y = batch
            x = x.to(device, non_blocking=True)
            x_sr = x_sr.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            delta = make_delta(x, x_sr, rect, use_abs=use_abs)

        logits = clf(delta).squeeze(1)
        y_pred.extend(logits.sigmoid().detach().cpu().tolist())
        y_true.extend(y.detach().cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, threshold)
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, y_true, y_pred


def main():
    args = parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    use_residual_cache = args.residual_cache_root is not None
    if use_residual_cache:
        ds = ResidualBinaryCacheDataset(args.residual_cache_root)
        print(f"Using residual cache for test: {args.residual_cache_root}")
    else:
        if args.data_root is None or args.sr_cache_root is None:
            raise ValueError("--data_root and --sr_cache_root are required when residual cache is not used.")
        ds = SRBinaryDataset(args.data_root, args.sr_cache_root, image_size=args.image_size)
        print(f"Using SR cache for test: {args.sr_cache_root}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

    rect = ResidualRectifierCNN(c_in=3, width=args.rect_width, depth=args.rect_depth).to(device)
    rect.load_state_dict(load_state_dict_clean(args.rect_ckpt, device))
    rect.eval()

    clf = SmallCNN(c_in=3, width=args.clf_width).to(device)
    clf.load_state_dict(load_state_dict_clean(args.clf_ckpt, device))
    clf.eval()

    ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, y_true, y_pred = evaluate(
        clf=clf,
        rect=rect,
        loader=dl,
        device=device,
        use_abs=args.use_abs,
        threshold=args.threshold,
        use_residual_cache=use_residual_cache,
    )

    print(f"AP: {ap:.6f}")
    print(f"ACC@{args.threshold:.2f}: real={r_acc0:.6f} fake={f_acc0:.6f} total={acc0:.6f}")
    print(f"BestThres: {best_thres:.6f}")
    print(f"ACC@Best: real={r_acc1:.6f} fake={f_acc1:.6f} total={acc1:.6f}")

    if args.save_npz:
        np.savez(args.save_npz, y_true=y_true, y_pred=y_pred)
        print(f"Saved predictions: {args.save_npz}")


if __name__ == "__main__":
    main()
