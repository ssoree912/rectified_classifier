import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.latent_rectify_dataset import LatentRectifyDataset
from dataset.sr_rectify_dataset import SRRectifyDataset
from models.latent_rectifier import (
    CLIPPenultimateEncoder,
    DEFAULT_TOKEN_MAP_CHANNELS,
    DEFAULT_TOKEN_MAP_GRID,
    LatentRectifierMLP,
    TokenMapRectifierCNN,
    clip_input_size,
)

try:
    import wandb
except ImportError:
    wandb = None


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train latent rectifier on CLIP latents")
    parser.add_argument("--img_dir", type=str, default=None, help="Directory with original training images")
    parser.add_argument("--sr_cache_root", type=str, default=None, help="Root of precomputed SR image cache")
    parser.add_argument("--clean_latent_root", type=str, default=None, help="Optional root of precomputed clean latents")
    parser.add_argument("--sr_latent_root", type=str, default=None, help="Optional root of precomputed SR latents")
    parser.add_argument("--save_path", type=str, default="latent_rectifier_latest.pth")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--resume_epoch", type=int, default=None, help="Resume from the checkpoint saved at the end of this 1-based epoch number")
    parser.add_argument("--arch", type=str, default="ViT-L/14", help="CLIP backbone used to define the latent space")
    parser.add_argument("--image_size", type=int, default=0, help="Input resize for paired images; <=0 uses the CLIP default size")
    parser.add_argument("--latent_kind", type=str, choices=["cls", "gap", "token_map"], default="token_map")
    parser.add_argument("--token_map_channels", type=int, default=DEFAULT_TOKEN_MAP_CHANNELS, help="Compressed token-map channel count; <=0 keeps the original token width")
    parser.add_argument("--token_map_grid", type=int, default=DEFAULT_TOKEN_MAP_GRID, help="Compressed token-map spatial size; <=0 keeps the original token grid")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden width for the latent rectifier")
    parser.add_argument("--depth", type=int, default=4, help="Rectifier depth")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss", type=str, choices=["l1", "mse"], default="l1")
    parser.add_argument("--save_every", type=int, default=1, help="Save an epoch checkpoint every N epochs")
    parser.add_argument("--save_every_steps", type=int, default=0, help="Update the latest checkpoint every N optimization steps; <=0 disables step checkpoints")
    parser.add_argument("--log_every", type=int, default=20, help="Log metrics every N optimization steps")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bar updates")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--path_contains",
        type=str,
        nargs="+",
        default=None,
        help="Only train on samples whose relative path contains one of these substrings.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights and Biases logging")
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


def checkpoint_path_for_epoch(save_path: Path, epoch: int) -> Path:
    stem = save_path.stem
    suffix = save_path.suffix or ".pth"
    return save_path.with_name(f"{stem}_epoch{epoch:03d}{suffix}")


def resolve_resume_path(args, save_path: Path):
    if args.resume_epoch is not None and args.resume is not None:
        raise ValueError("Use either --resume or --resume_epoch, not both.")
    if args.resume_epoch is not None:
        return checkpoint_path_for_epoch(save_path, args.resume_epoch).resolve()
    if args.resume:
        return Path(args.resume).resolve()
    return None


def peek_checkpoint_meta(resume_path: Path):
    if resume_path is None or not resume_path.exists():
        return {}
    checkpoint = torch.load(resume_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        return checkpoint
    return {}


def build_loader(dataset, args, device: str, epoch: int):
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
        generator=generator,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def build_checkpoint_state(rectifier, optimizer, scaler, args, next_epoch, next_step_in_epoch, global_step, wandb_run, feature_dim, grid_size, data_mode, rectifier_kind, latent_kind):
    state = {
        "model_state": rectifier.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "next_epoch": next_epoch,
        "next_step_in_epoch": next_step_in_epoch,
        "global_step": global_step,
        "feature_dim": feature_dim,
        "grid_size": grid_size,
        "data_mode": data_mode,
        "rectifier_kind": rectifier_kind,
        "latent_kind": latent_kind,
        "args": vars(args),
        "wandb_run_id": getattr(wandb_run, "id", None),
        "wandb_run_name": getattr(wandb_run, "name", None),
    }
    if scaler is not None and scaler.is_enabled():
        state["scaler_state"] = scaler.state_dict()
    return state


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_latest_checkpoint(save_path: Path, state):
    atomic_torch_save(state, save_path)


def save_epoch_checkpoint(save_path: Path, state, epoch: int):
    epoch_path = checkpoint_path_for_epoch(save_path, epoch)
    atomic_torch_save(state, epoch_path)
    return epoch_path


def load_resume_checkpoint(resume_path: Path, rectifier, optimizer, scaler, device: str):
    checkpoint = torch.load(resume_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        rectifier.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scaler_state = checkpoint.get("scaler_state")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        return checkpoint

    rectifier.load_state_dict(checkpoint, strict=True)
    return {
        "next_epoch": 0,
        "next_step_in_epoch": 0,
        "global_step": 0,
        "wandb_run_id": None,
        "wandb_run_name": None,
        "feature_dim": getattr(rectifier, "input_dim", None),
        "grid_size": 1,
        "data_mode": "image_pair",
        "rectifier_kind": "mlp",
        "latent_kind": "cls",
    }


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_timestamp(unix_seconds: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(unix_seconds))


def init_wandb(args, device: str, dataset_size: int, steps_per_epoch: int, resume_path=None, resume_id=None, run_name=None, feature_dim=None, grid_size=None, data_mode=None, rectifier_kind=None, latent_kind=None):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError(
            "wandb is not installed in the current environment. Install it or re-run without --wandb."
        )

    wandb_mode = resolve_wandb_mode(args.wandb_mode)
    wandb_dir = Path(args.save_path).resolve().parent / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_name,
        mode=wandb_mode,
        dir=str(wandb_dir),
        config={
            "img_dir": args.img_dir,
            "sr_cache_root": args.sr_cache_root,
            "clean_latent_root": args.clean_latent_root,
            "sr_latent_root": args.sr_latent_root,
            "arch": args.arch,
            "image_size": args.image_size,
            "latent_kind": latent_kind,
            "token_map_channels": args.token_map_channels,
            "token_map_grid": args.token_map_grid,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "lr": args.lr,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "dropout": args.dropout,
            "loss": args.loss,
            "save_every": args.save_every,
            "save_every_steps": args.save_every_steps,
            "path_contains": args.path_contains,
            "device": device,
            "dataset_size": dataset_size,
            "steps_per_epoch": steps_per_epoch,
            "resume_checkpoint": str(resume_path) if resume_path else None,
            "resume_epoch": args.resume_epoch,
            "disable_tqdm": args.disable_tqdm,
            "feature_dim": feature_dim,
            "grid_size": grid_size,
            "data_mode": data_mode,
            "rectifier_kind": rectifier_kind,
        },
    )
    if resume_id:
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "allow"

    return wandb.init(**init_kwargs)


def normalize_for_clip(x: torch.Tensor) -> torch.Tensor:
    mean = CLIP_MEAN.to(device=x.device, dtype=x.dtype)
    std = CLIP_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    return F.l1_loss(pred, target)


def resolve_training_data(args):
    using_cached_latents = args.clean_latent_root is not None or args.sr_latent_root is not None
    if using_cached_latents:
        if not args.clean_latent_root or not args.sr_latent_root:
            raise ValueError("Use both --clean_latent_root and --sr_latent_root together.")
        dataset = LatentRectifyDataset(
            clean_latent_root=args.clean_latent_root,
            sr_latent_root=args.sr_latent_root,
            include_path_contains=args.path_contains,
        )
        feature_dim = dataset.feature_dim
        grid_size = dataset.grid_size
        latent_kind = dataset.latent_kind
        rectifier_kind = "token_map_cnn" if dataset.is_spatial else "mlp"
        return dataset, None, feature_dim, grid_size, "cached_latent", latent_kind, rectifier_kind

    if not args.img_dir or not args.sr_cache_root:
        raise ValueError("Image-pair mode requires both --img_dir and --sr_cache_root.")
    if args.image_size <= 0:
        args.image_size = clip_input_size(args.arch)

    dataset = SRRectifyDataset(
        args.img_dir,
        image_size=args.image_size,
        sr_cache_root=args.sr_cache_root,
        include_path_contains=args.path_contains,
    )
    encoder = CLIPPenultimateEncoder(args.arch)
    if args.latent_kind == "token_map":
        feature_dim, grid_size = encoder.infer_token_map_shape(
            compress_channels=args.token_map_channels,
            compress_grid=args.token_map_grid,
        )
        rectifier_kind = "token_map_cnn"
    else:
        feature_dim = encoder.infer_feature_dim()
        grid_size = 1
        rectifier_kind = "mlp"
    return dataset, encoder, feature_dim, grid_size, "image_pair", args.latent_kind, rectifier_kind


def build_rectifier(rectifier_kind: str, feature_dim: int, hidden_dim: int, depth: int, dropout: float):
    if rectifier_kind == "token_map_cnn":
        return TokenMapRectifierCNN(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
    return LatentRectifierMLP(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    use_cuda = device.startswith("cuda")
    use_amp = use_cuda
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    save_path = Path(args.save_path).resolve()
    resume_path = resolve_resume_path(args, save_path)
    resume_meta = peek_checkpoint_meta(resume_path)

    dataset, encoder, feature_dim, grid_size, data_mode, latent_kind, rectifier_kind = resolve_training_data(args)
    if encoder is not None:
        encoder = encoder.to(device)
        encoder.eval()

    if isinstance(resume_meta, dict) and resume_meta:
        feature_dim = int(resume_meta.get("feature_dim", feature_dim))
        grid_size = int(resume_meta.get("grid_size", grid_size)) if resume_meta.get("grid_size") is not None else grid_size
        latent_kind = resume_meta.get("latent_kind", latent_kind)
        rectifier_kind = resume_meta.get("rectifier_kind", rectifier_kind)
        data_mode = resume_meta.get("data_mode", data_mode)

    base_loader = build_loader(dataset, args, device=device, epoch=0)
    steps_per_epoch = len(base_loader)
    del base_loader

    rectifier = build_rectifier(
        rectifier_kind=rectifier_kind,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(rectifier.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    resume_wandb_id = None
    resume_wandb_name = None

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = load_resume_checkpoint(resume_path, rectifier, optimizer, scaler, device=device)
        start_epoch = int(ckpt.get("next_epoch", 0))
        start_step_in_epoch = int(ckpt.get("next_step_in_epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        resume_wandb_id = ckpt.get("wandb_run_id")
        resume_wandb_name = ckpt.get("wandb_run_name")
        feature_dim = int(ckpt.get("feature_dim", feature_dim))
        grid_size = int(ckpt.get("grid_size", grid_size)) if ckpt.get("grid_size") is not None else grid_size
        data_mode = ckpt.get("data_mode", data_mode)
        rectifier_kind = ckpt.get("rectifier_kind", rectifier_kind)
        latent_kind = ckpt.get("latent_kind", latent_kind)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LatentRectifier] device={device}")
    print(f"[LatentRectifier] amp={use_amp}")
    print(f"[LatentRectifier] data_mode={data_mode}")
    print(f"[LatentRectifier] arch={args.arch}")
    print(f"[LatentRectifier] image_size={args.image_size}")
    print(f"[LatentRectifier] latent_kind={latent_kind}")
    if latent_kind == "token_map":
        print(f"[LatentRectifier] token_map_channels={args.token_map_channels}")
        print(f"[LatentRectifier] token_map_grid={args.token_map_grid}")
    print(f"[LatentRectifier] rectifier_kind={rectifier_kind}")
    print(f"[LatentRectifier] feature_dim={feature_dim}")
    print(f"[LatentRectifier] grid_size={grid_size}")
    if args.img_dir:
        print(f"[LatentRectifier] img_dir={Path(args.img_dir).resolve()}")
    if args.sr_cache_root:
        print(f"[LatentRectifier] sr_cache_root={Path(args.sr_cache_root).resolve()}")
    if args.clean_latent_root:
        print(f"[LatentRectifier] clean_latent_root={Path(args.clean_latent_root).resolve()}")
    if args.sr_latent_root:
        print(f"[LatentRectifier] sr_latent_root={Path(args.sr_latent_root).resolve()}")
    print(f"[LatentRectifier] dataset_size={len(dataset)}")
    print(f"[LatentRectifier] steps_per_epoch={steps_per_epoch}")
    print(f"[LatentRectifier] save_path={save_path}")
    print(f"[LatentRectifier] num_workers={args.num_workers} prefetch_factor={args.prefetch_factor}")
    print(f"[LatentRectifier] disable_tqdm={args.disable_tqdm}")
    if args.path_contains:
        print(f"[LatentRectifier] path_contains={args.path_contains}")
    if resume_path is not None:
        print(f"[LatentRectifier] resume={resume_path}")
        if args.resume_epoch is not None:
            print(f"[LatentRectifier] resume_epoch={args.resume_epoch}")
        print(f"[LatentRectifier] start_epoch={start_epoch} start_step_in_epoch={start_step_in_epoch} global_step={global_step}")

    wandb_run = init_wandb(
        args,
        device=device,
        dataset_size=len(dataset),
        steps_per_epoch=steps_per_epoch,
        resume_path=resume_path,
        resume_id=resume_wandb_id,
        run_name=resume_wandb_name,
        feature_dim=feature_dim,
        grid_size=grid_size,
        data_mode=data_mode,
        rectifier_kind=rectifier_kind,
        latent_kind=latent_kind,
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            rectifier.train()
            total_loss = 0.0
            processed_steps = 0
            epoch_start = time.time()
            step_offset = start_step_in_epoch if epoch == start_epoch else 0

            loader = build_loader(dataset, args, device=device, epoch=epoch)
            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                leave=not args.disable_tqdm,
                disable=args.disable_tqdm,
                mininterval=10.0,
            )
            for step, batch in enumerate(pbar, start=1):
                if step <= step_offset:
                    continue

                if data_mode == "cached_latent":
                    z, z_sr = batch
                    z = z.to(device, non_blocking=True)
                    z_sr = z_sr.to(device, non_blocking=True)
                else:
                    x, x_sr = batch
                    x = x.to(device, non_blocking=True)
                    x_sr = x_sr.to(device, non_blocking=True)
                    x_clip = normalize_for_clip(x)
                    x_sr_clip = normalize_for_clip(x_sr)
                    with torch.no_grad():
                        with autocast(enabled=use_amp):
                            z_all = encoder.encode_latent(
                                torch.cat([x_clip, x_sr_clip], dim=0),
                                latent_kind=latent_kind,
                                token_map_channels=args.token_map_channels,
                                token_map_grid=args.token_map_grid,
                            )
                            z, z_sr = z_all.chunk(2, dim=0)
                    z = z.float()
                    z_sr = z_sr.float()

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    z_hat = rectifier(z_sr)
                    loss = compute_loss(z_hat, z, args.loss)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                global_step += 1
                processed_steps += 1
                total_loss += loss.item()
                avg_so_far = total_loss / processed_steps

                if step % args.log_every == 0 or step == len(loader):
                    elapsed = time.time() - epoch_start
                    step_time_sec = elapsed / max(processed_steps, 1)
                    epoch_steps_left = len(loader) - step
                    epoch_eta_sec = step_time_sec * epoch_steps_left
                    total_steps_left = epoch_steps_left + max(args.epochs - (epoch + 1), 0) * steps_per_epoch
                    total_eta_sec = step_time_sec * total_steps_left
                    eta_finish = format_timestamp(time.time() + total_eta_sec)

                    if args.disable_tqdm:
                        print(
                            f"[Step] epoch={epoch + 1} step={step}/{len(loader)} "
                            f"global_step={global_step} loss={loss.item():.6f} avg={avg_so_far:.6f} "
                            f"step_time={step_time_sec:.3f}s eta_epoch={format_duration(epoch_eta_sec)} "
                            f"eta_total={format_duration(total_eta_sec)} finish={eta_finish}"
                        )
                    else:
                        pbar.set_postfix(
                            loss=f"{loss.item():.6f}",
                            avg=f"{avg_so_far:.6f}",
                            eta=format_duration(epoch_eta_sec),
                        )
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "train/loss_step": loss.item(),
                                "train/loss_avg": avg_so_far,
                                "train/epoch": epoch + 1,
                                "train/step_in_epoch": step,
                                "train/global_step": global_step,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/step_time_sec": step_time_sec,
                                "train/epoch_eta_sec": epoch_eta_sec,
                                "train/total_eta_sec": total_eta_sec,
                            },
                            step=global_step,
                        )

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    latest_state = build_checkpoint_state(
                        rectifier,
                        optimizer,
                        scaler,
                        args,
                        epoch,
                        step,
                        global_step,
                        wandb_run,
                        feature_dim,
                        grid_size,
                        data_mode,
                        rectifier_kind,
                        latent_kind,
                    )
                    save_latest_checkpoint(save_path, latest_state)
                    print(f"[Checkpoint] updated latest: {save_path} (epoch={epoch + 1}, step={step}, global_step={global_step})")

            avg_loss = total_loss / max(processed_steps, 1)
            epoch_time = time.time() - epoch_start
            print(f"[Epoch {epoch + 1}/{args.epochs}] {args.loss.upper()}: {avg_loss:.6f} | time={epoch_time:.1f}s")

            if wandb_run is not None:
                wandb.log(
                    {
                        "train/epoch_loss": avg_loss,
                        "train/epoch_time_sec": epoch_time,
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )

            latest_state = build_checkpoint_state(
                rectifier,
                optimizer,
                scaler,
                args,
                epoch + 1,
                0,
                global_step,
                wandb_run,
                feature_dim,
                grid_size,
                data_mode,
                rectifier_kind,
                latent_kind,
            )
            save_latest_checkpoint(save_path, latest_state)

            if (epoch + 1) % args.save_every == 0:
                epoch_ckpt = save_epoch_checkpoint(save_path, latest_state, epoch + 1)
                print(f"[Checkpoint] saved epoch checkpoint: {epoch_ckpt}")

        print(f"Training finished. Latest checkpoint: {save_path}")
        if wandb_run is not None:
            wandb.summary["latest_checkpoint"] = str(save_path)
            wandb.summary["dataset_size"] = len(dataset)
            wandb.summary["global_step"] = global_step
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
