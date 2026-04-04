import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


VALID_EXT = ".pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Pack mirrored clean/SR latent trees into shard memmaps")
    parser.add_argument("--clean_latent_root", type=str, required=True)
    parser.add_argument("--sr_latent_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("--shard_size", type=int, default=20000)
    parser.add_argument("--save_dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--path_contains",
        type=str,
        nargs="+",
        default=None,
        help="Only include samples whose relative path contains one of these substrings.",
    )
    return parser.parse_args()


def is_hidden_relative(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def extract_latent(obj):
    if torch.is_tensor(obj):
        latent = obj
    elif isinstance(obj, dict):
        for key in ("latent", "feature", "tensor"):
            if key in obj:
                latent = obj[key]
                break
        else:
            raise KeyError("Latent checkpoint dict must contain one of: latent, feature, tensor")
    else:
        raise TypeError(f"Unsupported latent object type: {type(obj)}")

    latent = latent.detach().cpu().float().contiguous()
    if latent.ndim not in (1, 3):
        raise ValueError(f"Expected latent vector (C,) or map (C,H,W), got shape={tuple(latent.shape)}")
    return latent


def load_meta(root: Path):
    meta_path = root / ".latent_cache_meta.pt"
    if meta_path.is_file():
        return torch.load(meta_path, map_location="cpu")
    return {}


def iter_clean_paths(split_root: Path, include_path_contains=None):
    include_path_contains = tuple(include_path_contains or [])
    for path in sorted(split_root.rglob(f"*{VALID_EXT}")):
        if not path.is_file():
            continue
        rel = path.relative_to(split_root)
        if path.name.startswith(".") or is_hidden_relative(rel):
            continue
        if include_path_contains and not any(token in str(rel) for token in include_path_contains):
            continue
        yield path, rel


def infer_label(rel: Path) -> int:
    for part in rel.parts:
        if part == "0_real":
            return 0
        if part == "1_fake":
            return 1
    raise ValueError(f"Unable to infer label from path: {rel}")


def open_memmap(path: Path, dtype: str, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def pack_split(args, split: str, clean_meta: dict, sr_meta: dict):
    clean_split_root = Path(args.clean_latent_root).resolve() / split
    sr_split_root = Path(args.sr_latent_root).resolve() / split
    if not clean_split_root.is_dir():
        raise FileNotFoundError(f"Clean split directory not found: {clean_split_root}")
    if not sr_split_root.is_dir():
        raise FileNotFoundError(f"SR split directory not found: {sr_split_root}")

    split_out = Path(args.output_root).resolve() / split
    split_out.mkdir(parents=True, exist_ok=True)

    samples = []
    missing = []
    for clean_path, rel in iter_clean_paths(clean_split_root, include_path_contains=args.path_contains):
        sr_path = sr_split_root / rel
        if not sr_path.is_file():
            missing.append(str(rel))
            continue
        samples.append((clean_path, sr_path, rel))

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"SR latent cache not found for {len(missing)} files under {sr_split_root}.\nExamples:\n{preview}"
        )
    if not samples:
        raise ValueError(f"No mirrored latent samples found for split={split}")

    probe = extract_latent(torch.load(samples[0][0], map_location="cpu"))
    latent_shape = tuple(probe.shape)
    feature_dim = int(probe.shape[0])
    grid_size = int(probe.shape[-1]) if probe.ndim == 3 else 1
    latent_kind = clean_meta.get("latent_kind") or sr_meta.get("latent_kind") or ("token_map" if probe.ndim == 3 else "cls")

    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    shards = []

    total = len(samples)
    print(f"[MemmapPack] split={split} total_count={total} latent_shape={latent_shape} save_dtype={args.save_dtype}")

    for shard_idx, start in enumerate(range(0, total, args.shard_size)):
        end = min(start + args.shard_size, total)
        shard_samples = samples[start:end]
        shard_count = len(shard_samples)

        clean_file = f"clean_shard_{shard_idx:03d}.npy"
        sr_file = f"sr_shard_{shard_idx:03d}.npy"
        label_file = f"label_shard_{shard_idx:03d}.npy"
        path_file = f"path_shard_{shard_idx:03d}.txt"

        clean_arr = open_memmap(split_out / clean_file, dtype=save_dtype, shape=(shard_count, *latent_shape))
        sr_arr = open_memmap(split_out / sr_file, dtype=save_dtype, shape=(shard_count, *latent_shape))
        label_arr = open_memmap(split_out / label_file, dtype=np.uint8, shape=(shard_count,))

        with (split_out / path_file).open("w", encoding="utf-8") as path_handle:
            for local_idx, (clean_path, sr_path, rel) in enumerate(tqdm(shard_samples, desc=f"{split} shard {shard_idx:03d}", leave=False)):
                clean_latent = extract_latent(torch.load(clean_path, map_location="cpu")).numpy().astype(save_dtype, copy=False)
                sr_latent = extract_latent(torch.load(sr_path, map_location="cpu")).numpy().astype(save_dtype, copy=False)
                clean_arr[local_idx] = clean_latent
                sr_arr[local_idx] = sr_latent
                label_arr[local_idx] = infer_label(rel)
                path_handle.write(f"{rel.as_posix()}\n")

        clean_arr.flush()
        sr_arr.flush()
        label_arr.flush()
        shards.append(
            {
                "index": shard_idx,
                "count": shard_count,
                "clean_file": clean_file,
                "sr_file": sr_file,
                "label_file": label_file,
                "path_file": path_file,
            }
        )

    manifest = {
        "split": split,
        "source_clean_root": str(clean_split_root),
        "source_sr_root": str(sr_split_root),
        "total_count": total,
        "feature_dim": feature_dim,
        "grid_size": grid_size,
        "latent_kind": latent_kind,
        "latent_shape": list(latent_shape),
        "save_dtype": args.save_dtype,
        "available_keys": ["clean", "sr"],
        "shard_size": args.shard_size,
        "shards": shards,
    }
    with (split_out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[MemmapPack] split={split} shards={len(shards)} output={split_out}")


def main():
    args = parse_args()
    clean_root = Path(args.clean_latent_root).resolve()
    sr_root = Path(args.sr_latent_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    clean_meta = load_meta(clean_root)
    sr_meta = load_meta(sr_root)
    for split in args.splits:
        pack_split(args, split=split, clean_meta=clean_meta, sr_meta=sr_meta)


if __name__ == "__main__":
    main()
