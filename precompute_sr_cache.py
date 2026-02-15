import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from models.sr_modules import BasicSRProcessor


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute SR(D(x)) cache for all images under input_root."
    )
    parser.add_argument("--input_root", type=str, required=True, help="Root folder to scan recursively")
    parser.add_argument("--output_root", type=str, required=True, help="Root folder to save SR images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sr_model_name", type=str, default="RealESRGAN_x4plus")
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--sr_tile", type=int, default=512)
    parser.add_argument("--save_ext", type=str, default="keep", choices=["keep", "png", "jpg"])
    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def list_images(root: Path):
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            paths.append(p)
    return sorted(paths)


def build_output_path(src_path: Path, input_root: Path, output_root: Path, save_ext: str):
    rel = src_path.relative_to(input_root)
    if save_ext == "keep":
        out_rel = rel
    elif save_ext == "png":
        out_rel = rel.with_suffix(".png")
    else:
        out_rel = rel.with_suffix(".jpg")
    return output_root / out_rel


def tensor_to_uint8_image(t: torch.Tensor):
    t = t.detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


def main():
    args = parse_args()
    device = resolve_device(args.device)
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    sr = BasicSRProcessor(
        scale=args.sr_scale,
        model_name=args.sr_model_name,
        device=device,
        tile=args.sr_tile,
    )

    paths = list_images(input_root)
    if args.max_items is not None:
        paths = paths[: args.max_items]
    if not paths:
        raise ValueError(f"No images found under: {input_root}")

    total = len(paths)
    saved = 0
    skipped = 0
    failed = 0

    print(f"[SR Cache] input_root={input_root}")
    print(f"[SR Cache] output_root={output_root}")
    print(f"[SR Cache] total_images={total}")

    for i, src_path in enumerate(paths, start=1):
        out_path = build_output_path(src_path, input_root, output_root, args.save_ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            skipped += 1
            if i % args.log_every == 0:
                print(f"[{i}/{total}] skipped={skipped} saved={saved} failed={failed}")
            continue

        try:
            img = Image.open(src_path).convert("RGB")
            x = TF.to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                x_sr = sr.sr_process(x)[0]

            sr_np = tensor_to_uint8_image(x_sr)
            sr_img = Image.fromarray(sr_np)

            if args.save_ext == "jpg":
                sr_img.save(out_path, quality=args.jpg_quality)
            elif args.save_ext == "keep" and out_path.suffix.lower() in {".jpg", ".jpeg"}:
                sr_img.save(out_path, quality=args.jpg_quality)
            else:
                sr_img.save(out_path)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {src_path}: {e}")

        if i % args.log_every == 0:
            print(f"[{i}/{total}] skipped={skipped} saved={saved} failed={failed}")

    print(f"[DONE] total={total} saved={saved} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
