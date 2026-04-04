import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.latent_memmap_dataset import LatentMemmapPairDataset
from models.latent_rectifier import LatentRectifierMLP, TokenMapRectifierCNN

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train a direct latent rectifier on memmap shard pairs")
    parser.add_argument("--bundle_root", type=str, required=True)
    parser.add_argument("--aux_key", type=str, default="sr")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss", type=str, choices=["l1", "mse"], default="l1")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
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


def build_rectifier(input_dim: int, is_spatial: bool, hidden_dim: int, depth: int, dropout: float):
    if is_spatial:
        return TokenMapRectifierCNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        ), "token_map_cnn"
    return LatentRectifierMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ), "mlp"


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_kind: str):
    if loss_kind == "mse":
        return F.mse_loss(pred, target)
    return F.l1_loss(pred, target)


def evaluate(model, loader, device: str, use_amp: bool, loss_kind: str):
    model.eval()
    losses = []
    with torch.no_grad():
        for z_clean, z_aux in loader:
            z_clean = z_clean.to(device, non_blocking=True)
            z_aux = z_aux.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                z_rect = model(z_aux)
                loss = compute_loss(z_rect, z_clean, loss_kind)
            losses.append(float(loss.item()))
    return {"loss": float(np.mean(losses)) if losses else 0.0}


def init_wandb(args, train_dataset, val_dataset, rectifier_kind, save_path: Path, device: str, steps_per_epoch: int, resume_id=None):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed in the current environment. Install it or re-run without --wandb.")

    wandb_dir = save_path.parent / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or save_path.parent.name,
        mode=resolve_wandb_mode(args.wandb_mode),
        dir=str(wandb_dir),
        config={
            "bundle_root": args.bundle_root,
            "aux_key": args.aux_key,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "dropout": args.dropout,
            "loss": args.loss,
            "rectifier_kind": rectifier_kind,
            "device": device,
            "feature_dim": train_dataset.feature_dim,
            "grid_size": train_dataset.grid_size,
            "is_spatial": train_dataset.is_spatial,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
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

    save_path = Path(args.save_path).resolve()
    latest_ckpt = save_path
    best_ckpt = save_path.with_name(f"{save_path.stem}_best{save_path.suffix or '.pth'}")

    train_dataset = LatentMemmapPairDataset(
        clean_root=args.bundle_root,
        aux_root=args.bundle_root,
        split=args.train_split,
        aux_key=args.aux_key,
        include_path_contains=args.path_contains,
        return_label=False,
        return_relpath=False,
    )
    val_dataset = LatentMemmapPairDataset(
        clean_root=args.bundle_root,
        aux_root=args.bundle_root,
        split=args.val_split,
        aux_key=args.aux_key,
        include_path_contains=args.path_contains,
        return_label=False,
        return_relpath=False,
    )

    train_loader = build_loader(train_dataset, args, device=device, shuffle=True)
    val_loader = build_loader(val_dataset, args, device=device, shuffle=False)

    model, rectifier_kind = build_rectifier(
        input_dim=train_dataset.feature_dim,
        is_spatial=train_dataset.is_spatial,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    resume_wandb_id = None

    resume_path = find_resume_path(args, latest_ckpt)
    if resume_path is not None and resume_path.exists():
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scaler=scaler, device=device)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        resume_wandb_id = ckpt.get("wandb_run_id")

    print(f"[DirectRectifier] device={device}")
    print(f"[DirectRectifier] bundle_root={Path(args.bundle_root).resolve()}")
    print(f"[DirectRectifier] aux_key={args.aux_key}")
    print(f"[DirectRectifier] train_size={len(train_dataset)} val_size={len(val_dataset)}")
    print(f"[DirectRectifier] feature_dim={train_dataset.feature_dim} grid_size={train_dataset.grid_size} is_spatial={train_dataset.is_spatial}")
    print(f"[DirectRectifier] rectifier_kind={rectifier_kind}")
    print(f"[DirectRectifier] steps_per_epoch={len(train_loader)}")
    print(f"[DirectRectifier] save_path={save_path}")
    print(f"[DirectRectifier] path_contains={args.path_contains}")

    wandb_run = init_wandb(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        rectifier_kind=rectifier_kind,
        save_path=save_path,
        device=device,
        steps_per_epoch=len(train_loader),
        resume_id=resume_wandb_id,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", mininterval=10.0)
        for step, (z_clean, z_aux) in enumerate(pbar, start=1):
            z_clean = z_clean.to(device, non_blocking=True)
            z_aux = z_aux.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                z_rect = model(z_aux)
                loss = compute_loss(z_rect, z_clean, args.loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.item())
            running_loss += loss_value
            global_step += 1
            pbar.set_postfix(loss=f"{loss_value:.6f}", avg=f"{running_loss / step:.6f}")

            if wandb_run is not None and (step % args.log_every == 0 or step == 1):
                wandb_run.log(
                    {
                        "global_step": global_step,
                        "epoch": epoch + step / max(len(train_loader), 1),
                        "train_step/loss": loss_value,
                        "train_step/avg_loss": running_loss / step,
                    }
                )

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device=device, use_amp=use_amp, loss_kind=args.loss)
        print(
            f"[Epoch {epoch + 1}] train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} best_val_loss={min(best_val_loss, val_metrics['loss']):.6f}"
        )

        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": min(best_val_loss, val_metrics["loss"]),
            "feature_dim": train_dataset.feature_dim,
            "grid_size": train_dataset.grid_size,
            "rectifier_kind": rectifier_kind,
            "args": vars(args),
            "wandb_run_id": getattr(wandb_run, "id", None),
        }
        save_checkpoint(latest_ckpt, state)
        if (epoch + 1) % args.save_every == 0:
            epoch_path = save_path.with_name(f"{save_path.stem}_epoch{epoch + 1:03d}{save_path.suffix or '.pth'}")
            save_checkpoint(epoch_path, state)

        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            state["best_val_loss"] = best_val_loss
            save_checkpoint(best_ckpt, state)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_epoch/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/best_loss": best_val_loss,
                }
            )
            wandb.summary["latest_checkpoint"] = str(latest_ckpt)
            wandb.summary["best_checkpoint"] = str(best_ckpt)

    print(f"[DirectRectifier] training complete latest={latest_ckpt} best={best_ckpt}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
