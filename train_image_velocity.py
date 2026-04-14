import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.sr_rectify_dataset import SRRectifyDataset
from models.velocity import StageVelocityUNet

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train image-space velocity rectifier: SR(x) -> x")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--sr_cache_root", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--integration_steps", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--path_contains", type=str, nargs="+", default=None)
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


def interpolate_images(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * x1


def velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return x1 - x0


@torch.no_grad()
def euler_transport(model, x0: torch.Tensor, num_steps: int = 4) -> torch.Tensor:
    x = x0
    batch_size = x.shape[0]
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t = torch.full((batch_size,), step / float(num_steps), device=x.device, dtype=torch.float32)
        v = model(x, t)
        x = x + dt * v
    return x


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


@torch.no_grad()
def evaluate(model, loader, device: str, use_amp: bool, integration_steps: int):
    model.eval()
    losses = []
    rectified_l1 = []
    for x_clean, x_sr in loader:
        x_clean = x_clean.to(device, non_blocking=True)
        x_sr = x_sr.to(device, non_blocking=True)
        t = torch.full((x_clean.shape[0],), 0.5, device=device, dtype=torch.float32)
        x_t = interpolate_images(x_sr, x_clean, t)
        target = velocity_target(x_sr, x_clean)
        with autocast(enabled=use_amp):
            pred = model(x_t, t)
            loss = F.mse_loss(pred, target)
        losses.append(float(loss.item()))
        x_rect = euler_transport(model, x_sr, num_steps=integration_steps)
        rectified_l1.append(float(F.l1_loss(x_rect, x_clean).item()))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "rectified_l1": float(np.mean(rectified_l1)) if rectified_l1 else 0.0,
    }


def init_wandb(args, train_dataset, val_dataset, save_path: Path, device: str, steps_per_epoch: int, resume_id=None):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed in the current environment.")

    wandb_dir = save_path.parent / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or save_path.parent.name,
        mode=resolve_wandb_mode(args.wandb_mode),
        dir=str(wandb_dir),
        config={
            "img_dir": args.img_dir,
            "sr_cache_root": args.sr_cache_root,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "num_res_blocks": args.num_res_blocks,
            "num_heads": args.num_heads,
            "integration_steps": args.integration_steps,
            "device": device,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "path_contains": args.path_contains,
            "steps_per_epoch": steps_per_epoch,
            "trainer": "image_velocity",
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

    train_dataset = SRRectifyDataset(
        args.img_dir,
        image_size=args.image_size,
        sr_cache_root=args.sr_cache_root,
        include_path_contains=args.path_contains,
    )
    val_root = str(Path(args.img_dir).resolve()).replace('/train', '/val')
    val_sr_root = str(Path(args.sr_cache_root).resolve()).replace('/train', '/val')
    val_dataset = SRRectifyDataset(
        val_root,
        image_size=args.image_size,
        sr_cache_root=val_sr_root,
        include_path_contains=args.path_contains,
    )

    train_loader = build_loader(train_dataset, args, device=device, shuffle=True)
    val_loader = build_loader(val_dataset, args, device=device, shuffle=False)

    model = StageVelocityUNet(
        c_in=3,
        c_hidden=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        num_heads=args.num_heads,
        residual=False,
    ).to(device)
    if device.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    resume_wandb_id = None

    if args.resume:
        resume_path = Path(args.resume).resolve()
    elif latest_ckpt.exists():
        resume_path = latest_ckpt
    else:
        resume_path = None

    if resume_path is not None and resume_path.exists():
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scaler=scaler, device=device)
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        global_step = int(ckpt.get('global_step', 0))
        best_val_loss = float(ckpt.get('best_val_loss', float('inf')))
        resume_wandb_id = ckpt.get('wandb_run_id')

    print(f"[ImageVelocity] device={device}")
    print(f"[ImageVelocity] img_dir={Path(args.img_dir).resolve()}")
    print(f"[ImageVelocity] sr_cache_root={Path(args.sr_cache_root).resolve()}")
    print(f"[ImageVelocity] train_size={len(train_dataset)} val_size={len(val_dataset)}")
    print(f"[ImageVelocity] image_size={args.image_size} batch_size={args.batch_size}")
    print(f"[ImageVelocity] steps_per_epoch={len(train_loader)} integration_steps={args.integration_steps}")
    print(f"[ImageVelocity] save_path={save_path}")
    if args.path_contains:
        print(f"[ImageVelocity] path_contains={args.path_contains}")
    if resume_path is not None and resume_path.exists():
        print(f"[ImageVelocity] resume={resume_path} start_epoch={start_epoch} global_step={global_step}")

    wandb_run = init_wandb(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=save_path,
        device=device,
        steps_per_epoch=len(train_loader),
        resume_id=resume_wandb_id,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", mininterval=10.0)
        for step, (x_clean, x_sr) in enumerate(pbar, start=1):
            x_clean = x_clean.to(device, non_blocking=True)
            x_sr = x_sr.to(device, non_blocking=True)
            if device.startswith("cuda"):
                x_clean = x_clean.to(memory_format=torch.channels_last)
                x_sr = x_sr.to(memory_format=torch.channels_last)
            t = torch.rand(x_clean.shape[0], device=device, dtype=torch.float32)
            x_t = interpolate_images(x_sr, x_clean, t)
            target = velocity_target(x_sr, x_clean)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred = model(x_t, t)
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
                    wandb_run.log({
                        'global_step': global_step,
                        'train_step/loss': float(loss.item()),
                        'train_step/avg_loss': avg_loss,
                    })

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(model, val_loader, device=device, use_amp=use_amp, integration_steps=args.integration_steps)
        val_loss = float(val_metrics['loss'])
        val_rectified_l1 = float(val_metrics['rectified_l1'])

        state = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict() if use_amp else None,
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': min(best_val_loss, val_loss),
            'args': vars(args),
            'wandb_run_id': getattr(wandb_run, 'id', None),
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
            wandb_run.log({
                'epoch': epoch + 1,
                'train_epoch/loss': train_loss,
                'val/loss': val_loss,
                'val/rectified_l1': val_rectified_l1,
                'val/best_loss': best_val_loss,
            })

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
