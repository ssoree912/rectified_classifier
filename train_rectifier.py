import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.sr_rectify_dataset import SRRectifyDataset
from models.velocity import RectifierUNet

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train SR rectifier: SR(D(x)) -> x")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with original training images")
    parser.add_argument("--sr_cache_root", type=str, required=True, help="Root of precomputed SR(D(x)) cache")
    parser.add_argument("--save_path", type=str, default="rectifier_latest.pth")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--resume_epoch", type=int, default=None, help="Resume from the checkpoint saved at the end of this 1-based epoch number")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=50)
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
        help="Only train on images whose path contains one of these substrings.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
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


def build_checkpoint_state(rectifier, optimizer, scaler, args, next_epoch, next_step_in_epoch, global_step, wandb_run):
    state = {
        "model_state": rectifier.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "next_epoch": next_epoch,
        "next_step_in_epoch": next_step_in_epoch,
        "global_step": global_step,
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
    }


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_timestamp(unix_seconds: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(unix_seconds))


def init_wandb(args, device: str, dataset_size: int, steps_per_epoch: int, resume_path=None, resume_id=None, run_name=None):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError(
            "wandb is not installed in the current environment. "
            "Install it or re-run without --wandb."
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
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "lr": args.lr,
            "epochs": args.epochs,
            "save_every": args.save_every,
            "save_every_steps": args.save_every_steps,
            "path_contains": args.path_contains,
            "device": device,
            "dataset_size": dataset_size,
            "steps_per_epoch": steps_per_epoch,
            "resume_checkpoint": str(resume_path) if resume_path else None,
            "resume_epoch": args.resume_epoch,
            "disable_tqdm": args.disable_tqdm,
        },
    )
    if resume_id:
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "allow"

    return wandb.init(**init_kwargs)


def main():
    args = parse_args()
    args.disable_tqdm = False
    device = resolve_device(args.device)
    use_cuda = device.startswith("cuda")
    use_amp = use_cuda
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    save_path = Path(args.save_path).resolve()
    resume_path = resolve_resume_path(args, save_path)

    dataset = SRRectifyDataset(
        args.img_dir,
        image_size=args.image_size,
        sr_cache_root=args.sr_cache_root,
        include_path_contains=args.path_contains,
    )

    base_loader = build_loader(dataset, args, device=device, epoch=0)
    steps_per_epoch = len(base_loader)
    del base_loader

    rectifier = RectifierUNet(c_in=3).to(device)
    if use_cuda:
        rectifier = rectifier.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(rectifier.parameters(), lr=args.lr)
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

    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Rectifier] device={device}")
    print(f"[Rectifier] amp={use_amp}")
    print(f"[Rectifier] img_dir={Path(args.img_dir).resolve()}")
    print(f"[Rectifier] sr_cache_root={Path(args.sr_cache_root).resolve()}")
    print(f"[Rectifier] dataset_size={len(dataset)}")
    print(f"[Rectifier] steps_per_epoch={steps_per_epoch}")
    print(f"[Rectifier] save_path={save_path}")
    print(f"[Rectifier] num_workers={args.num_workers} prefetch_factor={args.prefetch_factor}")
    print(f"[Rectifier] disable_tqdm={args.disable_tqdm}")
    if args.path_contains:
        print(f"[Rectifier] path_contains={args.path_contains}")
    if resume_path is not None:
        print(f"[Rectifier] resume={resume_path}")
        if args.resume_epoch is not None:
            print(f"[Rectifier] resume_epoch={args.resume_epoch}")
        print(f"[Rectifier] start_epoch={start_epoch} start_step_in_epoch={start_step_in_epoch} global_step={global_step}")

    wandb_run = init_wandb(
        args,
        device=device,
        dataset_size=len(dataset),
        steps_per_epoch=steps_per_epoch,
        resume_path=resume_path,
        resume_id=resume_wandb_id,
        run_name=resume_wandb_name,
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
            for step, (x, x_sr) in enumerate(pbar, start=1):
                if step <= step_offset:
                    continue

                if use_cuda:
                    x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                    x_sr = x_sr.to(device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    x = x.to(device, non_blocking=True)
                    x_sr = x_sr.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    x_hat = rectifier(x_sr)
                    loss = F.l1_loss(x_hat, x)

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
                        rectifier, optimizer, scaler, args, epoch, step, global_step, wandb_run
                    )
                    save_latest_checkpoint(save_path, latest_state)
                    print(f"[Checkpoint] updated latest: {save_path} (epoch={epoch + 1}, step={step}, global_step={global_step})")

            avg_loss = total_loss / max(processed_steps, 1)
            epoch_time = time.time() - epoch_start
            print(f"[Epoch {epoch + 1}/{args.epochs}] L1: {avg_loss:.6f} | time={epoch_time:.1f}s")

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
                rectifier, optimizer, scaler, args, epoch + 1, 0, global_step, wandb_run
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
