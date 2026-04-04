import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.latent_memmap_dataset import LatentMemmapPairDataset
from models.latent_velocity import (
    build_latent_velocity_model,
    euler_transport,
    interpolate_latents,
    velocity_target,
)

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train a time-conditioned latent velocity model on memmap shards")
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
    parser.add_argument("--t_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--model_kind", type=str, choices=["auto", "velocity_mlp", "token_map_velocity_cnn"], default="auto")
    parser.add_argument("--integration_steps", type=int, default=4)
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


def evaluate(model, loader, device: str, use_amp: bool, integration_steps: int):
    model.eval()
    losses = []
    rectified_l1 = []
    with torch.no_grad():
        for z_clean, z_aux in loader:
            z_clean = z_clean.to(device, non_blocking=True)
            z_aux = z_aux.to(device, non_blocking=True)
            t = torch.full((z_clean.shape[0],), 0.5, device=device, dtype=torch.float32)
            z_t = interpolate_latents(z_aux, z_clean, t)
            target = velocity_target(z_aux, z_clean)
            with autocast(enabled=use_amp):
                pred = model(z_t, t)
                loss = F.mse_loss(pred, target)
            losses.append(float(loss.item()))

            z_rect = euler_transport(model, z_aux, num_steps=integration_steps)
            rectified_l1.append(float(F.l1_loss(z_rect, z_clean).item()))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "rectified_l1": float(np.mean(rectified_l1)) if rectified_l1 else 0.0,
    }


def init_wandb(args, train_dataset, val_dataset, model_kind, save_path: Path, device: str, steps_per_epoch: int, resume_id=None):
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
            "t_dim": args.t_dim,
            "dropout": args.dropout,
            "model_kind": model_kind,
            "integration_steps": args.integration_steps,
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

    model, model_kind = build_latent_velocity_model(
        input_dim=train_dataset.feature_dim,
        is_spatial=train_dataset.is_spatial,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        t_dim=args.t_dim,
        dropout=args.dropout,
        model_kind=args.model_kind,
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

    print(f"[LatentVelocity] device={device}")
    print(f"[LatentVelocity] bundle_root={Path(args.bundle_root).resolve()}")
    print(f"[LatentVelocity] aux_key={args.aux_key}")
    print(f"[LatentVelocity] train_size={len(train_dataset)} val_size={len(val_dataset)}")
    print(f"[LatentVelocity] feature_dim={train_dataset.feature_dim} grid_size={train_dataset.grid_size} is_spatial={train_dataset.is_spatial}")
    print(f"[LatentVelocity] model_kind={model_kind}")
    print(f"[LatentVelocity] hidden_dim={args.hidden_dim} depth={args.depth} t_dim={args.t_dim} dropout={args.dropout}")
    print(f"[LatentVelocity] steps_per_epoch={len(train_loader)} integration_steps={args.integration_steps}")
    print(f"[LatentVelocity] save_path={save_path}")
    if args.path_contains:
        print(f"[LatentVelocity] path_contains={args.path_contains}")
    if resume_path is not None and resume_path.exists():
        print(f"[LatentVelocity] resume={resume_path} start_epoch={start_epoch} global_step={global_step}")

    wandb_run = init_wandb(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_kind=model_kind,
        save_path=save_path,
        device=device,
        steps_per_epoch=len(train_loader),
        resume_id=resume_wandb_id,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", mininterval=10.0)
        for step, (z_clean, z_aux) in enumerate(pbar, start=1):
            z_clean = z_clean.to(device, non_blocking=True)
            z_aux = z_aux.to(device, non_blocking=True)
            t = torch.rand(z_clean.shape[0], device=device, dtype=torch.float32)
            z_t = interpolate_latents(z_aux, z_clean, t)
            target = velocity_target(z_aux, z_clean)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred = model(z_t, t)
                loss = F.mse_loss(pred, target)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            global_step += 1
            epoch_losses.append(float(loss.item()))
            avg_loss = float(np.mean(epoch_losses))
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

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(model, val_loader, device=device, use_amp=use_amp, integration_steps=args.integration_steps)
        val_loss = float(val_metrics["loss"])
        val_rectified_l1 = float(val_metrics["rectified_l1"])

        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": min(best_val_loss, val_loss),
            "feature_dim": train_dataset.feature_dim,
            "grid_size": train_dataset.grid_size,
            "is_spatial": train_dataset.is_spatial,
            "model_kind": model_kind,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "t_dim": args.t_dim,
            "dropout": args.dropout,
            "integration_steps": args.integration_steps,
            "args": vars(args),
            "wandb_run_id": getattr(wandb_run, "id", None),
        }
        save_checkpoint(latest_ckpt, state)
        if (epoch + 1) % args.save_every == 0:
            epoch_path = save_path.with_name(f"{save_path.stem}_epoch{epoch + 1:03d}{save_path.suffix or '.pth'}")
            save_checkpoint(epoch_path, state)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt, state)

        print(
            f"[Epoch {epoch + 1}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_rectified_l1={val_rectified_l1:.6f} best_val_loss={best_val_loss:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_epoch/loss": train_loss,
                    "val/loss": val_loss,
                    "val/rectified_l1": val_rectified_l1,
                    "val/best_loss": best_val_loss,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
