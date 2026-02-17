import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.residual_binary_cache_dataset import ResidualBinaryCacheDataset
from dataset.sr_binary_dataset import SRBinaryDataset
from models.residual_rectifier import ResidualRectifierCNN
from models.delta_classifier import SmallCNN


def build_amp_tools(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        def autocast_ctx():
            return torch.amp.autocast("cuda", enabled=use_amp)

        return scaler, autocast_ctx

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def autocast_ctx():
        return torch.cuda.amp.autocast(enabled=use_amp)

    return scaler, autocast_ctx


def parse_args():
    p = argparse.ArgumentParser("Train classifier on delta = r - R(r)")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--sr_cache_root", type=str, default=None)
    p.add_argument("--residual_cache_root", type=str, default=None)
    p.add_argument("--val_data_root", type=str, nargs="*", default=None)
    p.add_argument("--val_sr_cache_root", type=str, default=None)
    p.add_argument("--val_residual_cache_root", type=str, default=None)
    p.add_argument("--rect_ckpt", type=str, required=True)
    p.add_argument("--save_path", type=str, default="clf_delta.pth")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_abs", action="store_true")
    p.add_argument("--rect_width", type=int, default=64)
    p.add_argument("--rect_depth", type=int, default=10)
    p.add_argument("--clf_width", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


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


def load_state_dict_clean(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


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
    y_true = []
    y_pred = []
    pbar = tqdm(loader, desc="[Eval]", leave=False)
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

    try:
        ap = average_precision_score(y_true, y_pred)
    except ValueError:
        ap = float("nan")

    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, threshold)
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


def main():
    args = parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = True

    use_residual_cache = args.residual_cache_root is not None
    if use_residual_cache:
        ds = ResidualBinaryCacheDataset(args.residual_cache_root)
        print(f"Using residual cache for train: {args.residual_cache_root}")
    else:
        if args.data_root is None or args.sr_cache_root is None:
            raise ValueError("--data_root and --sr_cache_root are required when residual cache is not used.")
        ds = SRBinaryDataset(args.data_root, args.sr_cache_root, image_size=args.image_size)
        print(f"Using SR cache for train: {args.sr_cache_root}")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

    val_loaders = []
    if use_residual_cache:
        if args.val_residual_cache_root is not None:
            vds = ResidualBinaryCacheDataset(args.val_residual_cache_root)
            vdl = DataLoader(
                vds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
            )
            val_name = args.val_data_root[0] if args.val_data_root else args.val_residual_cache_root
            val_loaders.append((val_name, vdl))
        elif args.val_data_root:
            raise ValueError(
                "--val_residual_cache_root is required for validation when --residual_cache_root is used."
            )
    else:
        if args.val_data_root:
            val_sr_cache_root = args.val_sr_cache_root or args.sr_cache_root
            for root in args.val_data_root:
                vds = SRBinaryDataset(root, val_sr_cache_root, image_size=args.image_size)
                vdl = DataLoader(
                    vds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device == "cuda"),
                    persistent_workers=(args.num_workers > 0),
                    prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
                )
                val_loaders.append((root, vdl))

    rect = ResidualRectifierCNN(c_in=3, width=args.rect_width, depth=args.rect_depth).to(device)
    rect.load_state_dict(load_state_dict_clean(args.rect_ckpt, device))
    rect.eval()
    for p in rect.parameters():
        p.requires_grad_(False)

    clf = SmallCNN(c_in=3, width=args.clf_width).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = args.amp and device == "cuda"
    scaler, autocast_ctx = build_amp_tools(use_amp)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_path = os.path.splitext(args.save_path)[0] + "_best.pth"
    best_acc = -1.0

    for ep in range(args.epochs):
        clf.train()
        total = 0.0
        pbar = tqdm(dl, desc=f"[Classifier] epoch {ep + 1}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            if use_residual_cache:
                r, y = batch
                r = r.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).float().unsqueeze(1)
                with torch.no_grad():
                    delta = make_delta_from_residual(r, rect, use_abs=args.use_abs)
            else:
                x, x_sr, y = batch
                x = x.to(device, non_blocking=True)
                x_sr = x_sr.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).float().unsqueeze(1)
                with torch.no_grad():
                    delta = make_delta(x, x_sr, rect, use_abs=args.use_abs)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = clf(delta)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total / step):.4f}")

        torch.save(clf.state_dict(), args.save_path)
        print(f"Saved: {args.save_path}")

        if val_loaders:
            acc_list = []
            ap_list = []
            b_acc_list = []
            thres_list = []
            for root, vdl in val_loaders:
                ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = evaluate(
                    clf=clf,
                    rect=rect,
                    loader=vdl,
                    device=device,
                    use_abs=args.use_abs,
                    threshold=args.threshold,
                    use_residual_cache=use_residual_cache,
                )
                print(
                    f"(Val on {root} @ epoch {ep + 1}) "
                    f"acc: {acc0:.4f}; ap: {ap:.4f}; "
                    f"r_acc0:{r_acc0:.4f}, f_acc0:{f_acc0:.4f}, "
                    f"r_acc1:{r_acc1:.4f}, f_acc1:{f_acc1:.4f}, "
                    f"acc1:{acc1:.4f}, best_thres:{best_thres:.4f}"
                )
                acc_list.append(acc0)
                ap_list.append(ap)
                b_acc_list.append(acc1)
                thres_list.append(best_thres)

            mean_acc = float(np.nanmean(acc_list))
            mean_ap = float(np.nanmean(ap_list))
            mean_b_acc = float(np.nanmean(b_acc_list))
            mean_thres = float(np.nanmean(thres_list))
            print(
                f"(average Val @ epoch {ep + 1}) "
                f"acc: {mean_acc:.4f}; ap: {mean_ap:.4f}; "
                f"b_acc: {mean_b_acc:.4f}; best_thres: {mean_thres:.4f}"
            )
            if mean_acc >= best_acc:
                best_acc = mean_acc
                torch.save(clf.state_dict(), best_path)
                print(f"Saved best: {best_path} (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
