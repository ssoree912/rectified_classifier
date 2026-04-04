import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset.latent_memmap_dataset import load_split_manifest, resolve_shard_file
from models.latent_rectifier import build_latent_rectifier_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute delta memmap shards from a trained direct latent rectifier")
    parser.add_argument("--bundle_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--rectifier_ckpt", type=str, required=True)
    parser.add_argument("--aux_key", type=str, default="sr")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--delta_mode",
        type=str,
        choices=["sr_minus_rectified", "orig_minus_rectified"],
        default="sr_minus_rectified",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def open_memmap(path: Path, dtype: str, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def build_manifest(source_manifest: dict, delta_mode: str, rectifier_ckpt: Path, save_dtype: str, shards: list):
    return {
        "split": source_manifest["split"],
        "source_bundle_root": str(source_manifest["_root"]),
        "total_count": int(source_manifest["total_count"]),
        "feature_dim": int(source_manifest["feature_dim"]),
        "grid_size": int(source_manifest.get("grid_size", 1)),
        "latent_kind": str(source_manifest.get("latent_kind", "cls")),
        "latent_shape": list(source_manifest["latent_shape"]),
        "save_dtype": save_dtype,
        "available_keys": ["delta"],
        "delta_mode": delta_mode,
        "rectifier_ckpt": str(rectifier_ckpt.resolve()),
        "shards": shards,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = device.startswith("cuda")
    if use_amp:
        torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.rectifier_ckpt, map_location="cpu")
    model, meta = build_latent_rectifier_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()

    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[DirectDelta] bundle_root={Path(args.bundle_root).resolve()}")
    print(f"[DirectDelta] output_root={output_root}")
    print(f"[DirectDelta] rectifier_ckpt={Path(args.rectifier_ckpt).resolve()}")
    print(f"[DirectDelta] device={device}")
    print(f"[DirectDelta] delta_mode={args.delta_mode}")
    print(f"[DirectDelta] rectifier_kind={meta['rectifier_kind']}")

    for split in args.splits:
        manifest = load_split_manifest(args.bundle_root, split)
        split_out = output_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        shard_entries = []

        print(f"[DirectDelta] split={split} total_count={manifest['total_count']}")
        for shard in manifest["shards"]:
            shard_idx = int(shard["index"])
            shard_count = int(shard["count"])
            delta_file = f"delta_shard_{shard_idx:03d}.npy"
            label_file = f"label_shard_{shard_idx:03d}.npy"
            path_file = f"path_shard_{shard_idx:03d}.txt"
            delta_path = split_out / delta_file

            if delta_path.exists() and not args.overwrite:
                shard_entries.append(
                    {
                        "index": shard_idx,
                        "count": shard_count,
                        "delta_file": delta_file,
                        "label_file": label_file,
                        "path_file": path_file,
                    }
                )
                continue

            clean_arr = np.load(resolve_shard_file(manifest, shard, "clean"), mmap_mode="r")
            aux_arr = np.load(resolve_shard_file(manifest, shard, args.aux_key), mmap_mode="r")
            delta_arr = open_memmap(delta_path, dtype=save_dtype, shape=(shard_count, *tuple(manifest["latent_shape"])))

            source_label = resolve_shard_file(manifest, shard, "label")
            source_path = resolve_shard_file(manifest, shard, "path")
            shutil.copyfile(source_label, split_out / label_file)
            shutil.copyfile(source_path, split_out / path_file)

            for start in tqdm(range(0, shard_count, args.batch_size), desc=f"{split} shard {shard_idx:03d}", leave=False):
                end = min(start + args.batch_size, shard_count)
                z_clean = torch.from_numpy(np.asarray(clean_arr[start:end])).float().to(device)
                z_aux = torch.from_numpy(np.asarray(aux_arr[start:end])).float().to(device)
                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            z_rect = model(z_aux)
                    else:
                        z_rect = model(z_aux)
                if args.delta_mode == "orig_minus_rectified":
                    delta = z_clean - z_rect
                else:
                    delta = z_aux - z_rect
                delta_arr[start:end] = delta.detach().cpu().numpy().astype(save_dtype, copy=False)

            delta_arr.flush()
            shard_entries.append(
                {
                    "index": shard_idx,
                    "count": shard_count,
                    "delta_file": delta_file,
                    "label_file": label_file,
                    "path_file": path_file,
                }
            )

        out_manifest = build_manifest(
            source_manifest=manifest,
            delta_mode=args.delta_mode,
            rectifier_ckpt=Path(args.rectifier_ckpt),
            save_dtype=args.save_dtype,
            shards=shard_entries,
        )
        with (split_out / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(out_manifest, handle, indent=2)
        print(f"[DirectDelta] split={split} shards={len(shard_entries)} output={split_out}")


if __name__ == "__main__":
    main()
