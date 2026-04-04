import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.latent_pair_classification_dataset import LatentPairClassificationDataset
from models.latent_classifier import build_latent_pair_classifier

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train attention classifier on cached latent pairs")
    parser.add_argument("--clean_latent_root", type=str, required=True)
    parser.add_argument("--aux_latent_root", type=str, required=True)
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
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["auto", "online", "offline", "disabled"],
        default="auto",
    )
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
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def checkpoint_path(save_dir: Path, name: str) -> Path:
    return save_dir / name


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


def resolve_resume_path(args, latest_ckpt: Path):
    if args.resume:
        return Path(args.resume).resolve()
    if latest_ckpt.exists():
        return latest_ckpt
    return None

def peek_checkpoint_meta(path: Path):
    if path is None or not path.exists():
        return {}
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        return checkpoint
    return {}


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


def evaluate(model, loader, device: str):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for z_clean, z_aux, labels, _ in loader:
            z_clean = z_clean.to(device, non_blocking=True)
            z_aux = z_aux.to(device, non_blocking=True)
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


def init_wandb(args, train_dataset, val_dataset, feature_dim, device, steps_per_epoch, save_dir, resume_id=None, classifier_kind=None):
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
            "clean_latent_root": args.clean_latent_root,
            "aux_latent_root": args.aux_latent_root,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "log_every": args.log_every,
            "save_every": args.save_every,
            "device": device,
            "feature_dim": feature_dim,
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
            "resume": args.resume,
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
    latest_ckpt = checkpoint_path(save_dir, "latest.pth")
    best_ckpt = checkpoint_path(save_dir, "best.pth")
    resume_path = resolve_resume_path(args, latest_ckpt)
    resume_meta = peek_checkpoint_meta(resume_path)
    resume_args = resume_meta.get("args", {}) if isinstance(resume_meta, dict) else {}

    train_dataset = LatentPairClassificationDataset(
        clean_latent_root=args.clean_latent_root,
        aux_latent_root=args.aux_latent_root,
        split=args.train_split,
        include_path_contains=args.path_contains,
    )
    val_dataset = LatentPairClassificationDataset(
        clean_latent_root=args.clean_latent_root,
        aux_latent_root=args.aux_latent_root,
        split=args.val_split,
        include_path_contains=args.path_contains,
    )

    train_loader = build_loader(train_dataset, args, device=device, shuffle=True)
    val_loader = build_loader(val_dataset, args, device=device, shuffle=False)

    classifier_kind_arg = resume_meta.get("classifier_kind") or resume_args.get("classifier_kind") or args.classifier_kind
    map_hidden_dim = int(resume_args.get("map_hidden_dim", args.map_hidden_dim))
    map_depth = int(resume_args.get("map_depth", args.map_depth))
    map_dropout = float(resume_args.get("map_dropout", args.map_dropout))
    model, classifier_kind = build_latent_pair_classifier(
        input_dim=train_dataset.feature_dim,
        is_spatial=train_dataset.is_spatial,
        classifier_kind=classifier_kind_arg,
        map_hidden_dim=map_hidden_dim,
        map_depth=map_depth,
        map_dropout=map_dropout,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
    criterion = torch.nn.BCEWithLogitsLoss()

    start_epoch = 0
    global_step = 0
    best_ap = -1.0
    resume_wandb_id = None
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scaler=scaler, device=device)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_ap = float(ckpt.get("best_ap", -1.0))
        resume_wandb_id = ckpt.get("wandb_run_id")

    print(f"[LatentClassifier] device={device}")
    print(f"[LatentClassifier] clean_latent_root={Path(args.clean_latent_root).resolve()}")
    print(f"[LatentClassifier] aux_latent_root={Path(args.aux_latent_root).resolve()}")
    print(f"[LatentClassifier] train_split={args.train_split} val_split={args.val_split}")
    print(f"[LatentClassifier] train_size={len(train_dataset)} val_size={len(val_dataset)}")
    print(f"[LatentClassifier] feature_dim={train_dataset.feature_dim} is_spatial={train_dataset.is_spatial}")
    print(f"[LatentClassifier] grid_size={train_dataset.grid_size}")
    print(f"[LatentClassifier] classifier_kind={classifier_kind}")
    if classifier_kind in {"map_cnn", "map_attention"}: 
        print(f"[LatentClassifier] map_hidden_dim={map_hidden_dim} map_depth={map_depth} map_dropout={map_dropout}")
    print(f"[LatentClassifier] train_class_counts={train_dataset.class_counts}")
    print(f"[LatentClassifier] val_class_counts={val_dataset.class_counts}")
    print(f"[LatentClassifier] steps_per_epoch={len(train_loader)}")
    print(f"[LatentClassifier] save_dir={save_dir}")
    print(f"[LatentClassifier] epochs={args.epochs}")
    if resume_path is not None:
        print(f"[LatentClassifier] resume={resume_path} start_epoch={start_epoch} global_step={global_step}")

    wandb_run = init_wandb(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        feature_dim=train_dataset.feature_dim,
        device=device,
        steps_per_epoch=len(train_loader),
        save_dir=save_dir,
        resume_id=resume_wandb_id,
        classifier_kind=classifier_kind,
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_start = time.time()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", mininterval=10.0)
            for step, (z_clean, z_aux, labels, _) in enumerate(pbar, start=1):
                z_clean = z_clean.to(device, non_blocking=True)
                z_aux = z_aux.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    logits = model(z_clean, z_aux).squeeze(1)
                    loss = criterion(logits, labels)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                global_step += 1
                running_loss += float(loss.item())
                avg_loss = running_loss / step
                if step % args.log_every == 0 or step == len(train_loader):
                    elapsed = time.time() - epoch_start
                    step_time = elapsed / step
                    eta = step_time * (len(train_loader) - step)
                    pbar.set_postfix(loss=f"{loss.item():.6f}", avg=f"{avg_loss:.6f}", eta=f"{eta/60:.1f}m")
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "train_step/loss": float(loss.item()),
                                "train_step/loss_avg": avg_loss,
                                "train_step/lr": optimizer.param_groups[0]["lr"],
                                "train_step/time_sec": step_time,
                            },
                            step=global_step,
                        )

            train_loss = running_loss / max(len(train_loader), 1)
            metrics = evaluate(model, val_loader, device=device)
            epoch_time = time.time() - epoch_start
            improved = metrics["ap"] >= best_ap
            if improved:
                best_ap = metrics["ap"]

            print(
                f"[Epoch {epoch + 1}/{args.epochs}] loss={train_loss:.6f} "
                f"val_ap={metrics['ap']:.6f} val_acc={metrics['acc']:.6f} "
                f"val_real_acc={metrics['real_acc']:.6f} val_fake_acc={metrics['fake_acc']:.6f} "
                f"val_bal_acc={metrics['balanced_acc']:.6f} val_best_acc={metrics['best_acc']:.6f} "
                f"val_best_real_acc={metrics['best_real_acc']:.6f} val_best_fake_acc={metrics['best_fake_acc']:.6f} "
                f"val_best_bal_acc={metrics['best_balanced_acc']:.6f} th={metrics['best_threshold']:.6f} "
                f"time={epoch_time:.1f}s"
            )

            ckpt_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "epoch": epoch,
                "global_step": global_step,
                "best_ap": best_ap,
                "feature_dim": train_dataset.feature_dim,
                "grid_size": train_dataset.grid_size,
                "is_spatial": train_dataset.is_spatial,
                "classifier_kind": classifier_kind,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "clean_latent_root": args.clean_latent_root,
                "aux_latent_root": args.aux_latent_root,
                "wandb_run_id": getattr(wandb_run, "id", None),
                "args": vars(args),
            }
            save_checkpoint(latest_ckpt, ckpt_state)
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(checkpoint_path(save_dir, f"epoch_{epoch + 1:03d}.pth"), ckpt_state)
            if improved:
                save_checkpoint(best_ckpt, ckpt_state)

            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_epoch/loss": train_loss,
                        "train_epoch/time_sec": epoch_time,
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
                        "val/best_ap_so_far": best_ap,
                    },
                    step=global_step,
                )
                wandb.summary["latest_checkpoint"] = str(latest_ckpt)
                wandb.summary["best_checkpoint"] = str(best_ckpt)
                wandb.summary["best_ap"] = best_ap
                wandb.summary["epoch"] = epoch + 1
                wandb.summary["global_step"] = global_step

    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
