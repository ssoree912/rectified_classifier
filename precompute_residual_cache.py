import argparse
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute residual cache r = x_sr - x and save as .pt tensors."
    )
    parser.add_argument("--input_root", type=str, required=True, help="Original image root")
    parser.add_argument("--sr_root", type=str, required=True, help="SR cache root")
    parser.add_argument("--output_root", type=str, required=True, help="Residual cache root")
    parser.add_argument(
        "--extra_job",
        nargs=3,
        action="append",
        default=[],
        metavar=("INPUT_ROOT", "SR_ROOT", "OUTPUT_ROOT"),
        help="Optional extra jobs. Can be repeated.",
    )
    parser.add_argument("--image_size", type=int, default=256, help="Resize target; <=0 disables resize")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--use_abs", action="store_true", help="Save |x_sr - x| instead of signed residual")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=1000)
    return parser.parse_args()


def list_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS)


def resolve_pair_path(src_path: Path, input_root: Path, sr_root: Path):
    rel = src_path.relative_to(input_root)
    candidates = [
        sr_root / rel,
        sr_root / input_root.name / rel,
    ]
    for p in candidates:
        if p.exists():
            return p
        for ext in VALID_EXTS:
            p2 = p.with_suffix(ext)
            if p2.exists():
                return p2
    return None


def build_output_path(src_path: Path, input_root: Path, output_root: Path):
    rel = src_path.relative_to(input_root)
    return (output_root / rel).with_suffix(".pt")


def load_tensor(path: Path, resize_tf=None):
    img = Image.open(path).convert("RGB")
    if resize_tf is not None:
        img = resize_tf(img)
    return TF.to_tensor(img)


def run_job(input_root: Path, sr_root: Path, output_root: Path, args):
    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    if not sr_root.exists():
        raise FileNotFoundError(f"sr_root does not exist: {sr_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    resize_tf = None
    if args.image_size and args.image_size > 0:
        resize_tf = T.Resize((args.image_size, args.image_size))

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    paths = list_images(input_root)
    if args.max_items is not None:
        paths = paths[: args.max_items]
    if not paths:
        raise ValueError(f"No images found under: {input_root}")

    total = len(paths)
    saved = 0
    skipped = 0
    failed = 0

    print(f"[Residual Cache] input_root={input_root}")
    print(f"[Residual Cache] sr_root={sr_root}")
    print(f"[Residual Cache] output_root={output_root}")
    print(f"[Residual Cache] total_images={total}")

    for i, src_path in enumerate(paths, start=1):
        out_path = build_output_path(src_path, input_root, output_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            skipped += 1
            if i % args.log_every == 0:
                print(f"[{i}/{total}] skipped={skipped} saved={saved} failed={failed}")
            continue

        try:
            sr_path = resolve_pair_path(src_path, input_root, sr_root)
            if sr_path is None:
                raise FileNotFoundError(f"SR file missing for {src_path}")

            x = load_tensor(src_path, resize_tf=resize_tf)
            x_sr = load_tensor(sr_path, resize_tf=resize_tf)

            if x.shape != x_sr.shape:
                raise ValueError(f"Shape mismatch: x={tuple(x.shape)} x_sr={tuple(x_sr.shape)}")

            residual = x_sr - x
            if args.use_abs:
                residual = residual.abs()

            residual = residual.to(dtype=dtype).contiguous()
            torch.save(residual, out_path)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {src_path}: {e}")

        if i % args.log_every == 0:
            print(f"[{i}/{total}] skipped={skipped} saved={saved} failed={failed}")

    print(f"[DONE] total={total} saved={saved} skipped={skipped} failed={failed}")


def main():
    args = parse_args()
    jobs = [(args.input_root, args.sr_root, args.output_root)] + list(args.extra_job)
    for idx, (input_root, sr_root, output_root) in enumerate(jobs, start=1):
        print(f"\n=== Job {idx}/{len(jobs)} ===")
        run_job(
            input_root=Path(input_root).resolve(),
            sr_root=Path(sr_root).resolve(),
            output_root=Path(output_root).resolve(),
            args=args,
        )


if __name__ == "__main__":
    main()
