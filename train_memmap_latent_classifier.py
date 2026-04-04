import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.latent_memmap_dataset import LatentMemmapPairDataset
from models.latent_classifier import build_latent_pair_classifier

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train a latent classifier from memmap shard pairs")
    parser.add_argument("--clean_root", type=str, required=True)
    parser.add_argument("--aux_root", type=str, required=True)
    parser.add_argument("--aux_key", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--classifier_kind", type=str, choices=["auto", "vector_attention", "map_cnn", "map_attention"], default="auto")
    parser.add_argument("--map_hidden_dim", type=int, default=128)
    parser.add_argument("--map_depth", type=int, default=4)
    parser.add_argument("--map_dropout", type=float, default=0.0)
    parser.add_argument(
        "--path_contains",
        type=str,
        nargs="+",
        default=None,
        help="Only include samples whose relative path contains one of these substrings.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rectified-classifier")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, choices=["auto", "online", "offline", "disabled"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def resolve_wandb_mode(mode_arg: str) -> str:
    if mode_arg != "auto":
        return mode_arg
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    return "offline"


def build_loader(dataset, args, device: str, shuffle: bool):
    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def save_checkpoint(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def find_resume_path(args, latest_path: Path):
    if args.resume:
        return Path(args.resume).resolve()
    if latest_path.exists():
        return latest_path
    return None


def find_best_threshold(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.5
    best_acc = -1.0
    best_thres = 0.5
    for thres in y_pred:
        pred = (y_pred >= thres).astype(np.int64)
        acc = float((pred == y_true).mean())
        if acc >= best_acc:
            best_acc = acc
            best_thres = float(thres)
    return best_thres


def calculate_split_acc(y_true, y_pred, threshold: float):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pred = (y_pred >= threshold).astype(np.int64)
    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_acc = float((pred[real_mask] == 0).mean()) if real_mask.any() else 0.0
    fake_acc = float((pred[fake_mask] == 1).mean()) if fake_mask.any() else 0.0
    acc = float((pred == y_true).mean())
    balanced_acc = 0.5 * (real_acc + fake_acc)
    return real_acc, fake_acc, acc, balanced_acc


def evaluate(model, loader, device: str, use_amp: bool):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for z_clean, z_aux, labels in loader:
            z_clean = z_clean.to(device, non_blocking=True)
            z_aux = z_aux.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(z_clean, z_aux)
            probs = logits.sigmoid().flatten().cpu().numpy()
            y_pred.extend(probs.tolist())
            y_true.extend(labels.numpy().astype(np.int64).tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ap = float(average_precision_score(y_true, y_pred))
    real_acc05, fake_acc05, acc05, balanced_acc05 = calculate_split_acc(y_true, y_pred, 0.5)
    best_thres = find_best_threshold(y_true, y_pred)
    real_best_acc, fake_best_acc, best_acc, balanced_best_acc = calculate_split_acc(y_true, y_pred, best_thres)
    return {
        "ap": ap,
        "acc": acc05,
        "real_acc": real_acc05,
        "fake_acc": fake_acc05,
        "balanced_acc": balanced_acc05,
        "best_acc": best_acc,
        "best_real_acc": real_best_acc,
        "best_fake_acc": fake_best_acc,
        "best_balanced_acc": balanced_best_acc,
        "best_threshold": best_thres,
    }


def init_wandb(args, train_dataset, val_dataset, classifier_kind, save_dir: Path, device: str, steps_per_epoch: int, resume_id=None):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed in the current environment. Install it or re-run without --wandb.")

    wandb_dir = save_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or save_dir.name,
        mode=resolve_wandb_mode(args.wandb_mode),
        dir=str(wandb_dir),
        config={
            "clean_root": args.clean_root,
            "aux_root": args.aux_root,
            "aux_key": args.aux_key,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "device": device,
            "feature_dim": train_dataset.feature_dim,
            "grid_size": train_dataset.grid_size,
            "is_spatial": train_dataset.is_spatial,
            "classifier_kind": classifier_kind,
            "map_hidden_dim": args.map_hidden_dim,
            "map_depth": args.map_depth,
            "map_dropout": args.map_dropout,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "train_class_counts": train_dataset.class_counts,
            "val_class_counts": val_dataset.class_counts,
            "path_contains": args.path_contains,
            "steps_per_epoch": steps_per_epoch,
        },
    )
    if resume_id:
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "allow"
    run = wandb.init(**init_kwargs)
    wandb.define_metric("epoch")
    wandb.define_metric("global_step")
    wandb.define_metric("train_step/*", step_metric="global_step")
    wandb.define_metric("train_epoch/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    return run


def main():
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = device.startswith("cuda")
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = save_dir / "latest.pth"
    best_ckpt = save_dir / "best.pth"

    train_dataset = LatentMemmapPairDataset(
        clean_root=args.clean_root,
        aux_root=args.aux_root,
        split=args.train_split,
        aux_key=args.aux_key,
        include_path_contains=args.path_contains,
        return_label=True,
        return_relpath=False,
    )
    val_dataset = LatentMemmapPairDataset(
        clean_root=args.clean_root,
        aux_root=args.aux_root,
        split=args.val_split,
        aux_key=args.aux_key,
        include_path_contains=args.path_contains,
        return_label=True,
        return_relpath=False,
    )

    train_loader = build_loader(train_dataset, args, device=device, shuffle=True)
    val_loader = build_loader(val_dataset, args, device=device, shuffle=False)

    model, classifier_kind = build_latent_pair_classifier(
        input_dim=train_dataset.feature_dim,
        is_spatial=train_dataset.is_spatial,
        classifier_kind=args.classifier_kind,
        map_hidden_dim=args.map_hidden_dim,
        map_depth=args.map_depth,
        map_dropout=args.map_dropout,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
    criterion = torch.nn.BCEWithLogitsLoss()

    start_epoch = 0
    global_step = 0
    best_ap = -1.0
    resume_wandb_id = None
    resume_path = find_resume_path(args, latest_ckpt)
    if resume_path is not None and resume_path.exists():
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scaler=scaler, device=device)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_ap = float(ckpt.get("best_ap", -1.0))
        resume_wandb_id = ckpt.get("wandb_run_id")

    print(f"[MemmapLatentClassifier] device={device}")
    print(f"[MemmapLatentClassifier] clean_root={Path(args.clean_root).resolve()}")
    print(f"[MemmapLatentClassifier] aux_root={Path(args.aux_root).resolve()} aux_key={args.aux_key}")
    print(f"[MemmapLatentClassifier] train_size={len(train_dataset)} val_size={len(val_dataset)}")
    print(f"[MemmapLatentClassifier] feature_dim={train_dataset.feature_dim} grid_size={train_dataset.grid_size} is_spatial={train_dataset.is_spatial}")
    print(f"[MemmapLatentClassifier] classifier_kind={classifier_kind}")
    print(f"[MemmapLatentClassifier] steps_per_epoch={len(train_loader)}")
    print(f"[MemmapLatentClassifier] save_dir={save_dir}")
    if args.path_contains:
        print(f"[MemmapLatentClassifier] path_contains={args.path_contains}")
    if resume_path is not None and resume_path.exists():
        print(f"[MemmapLatentClassifier] resume={resume_path} start_epoch={start_epoch} global_step={global_step}")

    wandb_run = init_wandb(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        classifier_kind=classifier_kind,
        save_dir=save_dir,
        device=device,
        steps_per_epoch=len(train_loader),
        resume_id=resume_wandb_id,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", mininterval=10.0)
        for step, (z_clean, z_aux, labels) in enumerate(pbar, start=1):
            z_clean = z_clean.to(device, non_blocking=True)
            z_aux = z_aux.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(z_clean, z_aux).flatten()
                loss = criterion(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            global_step += 1
            running_losses.append(float(loss.item()))
            avg_loss = float(np.mean(running_losses))
            if step % args.log_every == 0 or step == len(train_loader):
                pbar.set_postfix(loss=f"{loss.item():.6f}", avg=f"{avg_loss:.6f}")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "global_step": global_step,
                            "train_step/loss": float(loss.item()),
                            "train_step/avg_loss": avg_loss,
                        }
                    )

        train_loss = float(np.mean(running_losses)) if running_losses else 0.0
        metrics = evaluate(model, val_loader, device=device, use_amp=use_amp)

        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_ap": max(best_ap, metrics["ap"]),
            "feature_dim": train_dataset.feature_dim,
            "grid_size": train_dataset.grid_size,
            "is_spatial": train_dataset.is_spatial,
            "classifier_kind": classifier_kind,
            "aux_key": args.aux_key,
            "args": vars(args),
            "wandb_run_id": getattr(wandb_run, "id", None),
        }
        save_checkpoint(latest_ckpt, state)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(save_dir / f"epoch_{epoch + 1:03d}.pth", state)
        if metrics["ap"] >= best_ap:
            best_ap = metrics["ap"]
            save_checkpoint(best_ckpt, state)

        print(
            f"[Epoch {epoch + 1}] train_loss={train_loss:.6f} val_ap={metrics['ap']:.6f} "
            f"val_acc={metrics['acc']:.6f} val_real_acc={metrics['real_acc']:.6f} "
            f"val_fake_acc={metrics['fake_acc']:.6f} val_balanced_acc={metrics['balanced_acc']:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_epoch/loss": train_loss,
                    "val/ap": metrics["ap"],
                    "val/acc": metrics["acc"],
                    "val/real_acc": metrics["real_acc"],
                    "val/fake_acc": metrics["fake_acc"],
                    "val/balanced_acc": metrics["balanced_acc"],
                    "val/best_acc": metrics["best_acc"],
                    "val/best_real_acc": metrics["best_real_acc"],
                    "val/best_fake_acc": metrics["best_fake_acc"],
                    "val/best_balanced_acc": metrics["best_balanced_acc"],
                    "val/best_threshold": metrics["best_threshold"],
                    "val/best_ap": best_ap,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
