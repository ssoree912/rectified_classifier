import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms as T

from models.latent_rectifier import (
    CLIPPenultimateEncoder,
    DEFAULT_TOKEN_MAP_CHANNELS,
    DEFAULT_TOKEN_MAP_GRID,
    clip_input_size,
)


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CLIP latent cache for an image tree")
    parser.add_argument("--img_dir", type=str, required=True, help="Root directory with source images")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory to store latent `.pt` files")
    parser.add_argument("--arch", type=str, default="ViT-L/14", help="CLIP backbone used to define the latent space")
    parser.add_argument("--image_size", type=int, default=0, help="Resize before CLIP encoding; <=0 uses the CLIP default size")
    parser.add_argument("--latent_kind", type=str, choices=["cls", "gap", "token_map"], default="token_map")
    parser.add_argument(
        "--token_map_channels",
        type=int,
        default=DEFAULT_TOKEN_MAP_CHANNELS,
        help="Compressed channel count for token-map caching; <=0 keeps the original channel count",
    )
    parser.add_argument(
        "--token_map_grid",
        type=int,
        default=DEFAULT_TOKEN_MAP_GRID,
        help="Compressed spatial grid size for token-map caching; <=0 keeps the original grid size",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples whose latent cache file already exists")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing latent files")
    parser.add_argument(
        "--path_contains",
        type=str,
        nargs="+",
        default=None,
        help="Only include images whose path contains one of these substrings.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def normalize_for_clip(x: torch.Tensor) -> torch.Tensor:
    mean = CLIP_MEAN.to(device=x.device, dtype=x.dtype)
    std = CLIP_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def latent_path_for_relative(output_root: Path, rel: Path) -> Path:
    return (output_root / rel).with_suffix(".pt")


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def normalize_optional_size(value: int):
    value = int(value)
    return value if value > 0 else None


class ImagePathDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        output_root: str,
        image_size: int,
        include_path_contains=None,
        exclude_hidden: bool = True,
        skip_existing: bool = False,
    ):
        self.img_root = Path(img_dir).resolve()
        self.output_root = Path(output_root).resolve()
        self.include_path_contains = tuple(include_path_contains or [])
        self.exclude_hidden = exclude_hidden
        self.skip_existing = skip_existing
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.samples = []
        for path in sorted(self.img_root.rglob("*")):
            if not self._is_valid_image_path(path):
                continue
            rel = path.relative_to(self.img_root)
            out_path = latent_path_for_relative(self.output_root, rel)
            if self.skip_existing and out_path.is_file():
                continue
            self.samples.append((path, rel))

        if not self.samples:
            raise ValueError(f"No images found to encode under: {self.img_root}")

    def __len__(self):
        return len(self.samples)

    def _is_hidden_path(self, path: Path) -> bool:
        rel = path.relative_to(self.img_root)
        return any(part.startswith(".") for part in rel.parts)

    def _matches_include_filter(self, path: Path) -> bool:
        if not self.include_path_contains:
            return True
        path_str = str(path)
        return any(token in path_str for token in self.include_path_contains)

    def _is_valid_image_path(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if path.suffix.lower() not in VALID_EXTS:
            return False
        if self.exclude_hidden and self._is_hidden_path(path):
            return False
        return self._matches_include_filter(path)

    def __getitem__(self, idx):
        img_path, rel = self.samples[idx]
        with Image.open(img_path) as img:
            x = self.transform(img.convert("RGB"))
        return x, str(rel)


def encode_batch(encoder: CLIPPenultimateEncoder, x_clip: torch.Tensor, args):
    return encoder.encode_latent(
        x_clip,
        latent_kind=args.latent_kind,
        token_map_channels=normalize_optional_size(args.token_map_channels),
        token_map_grid=normalize_optional_size(args.token_map_grid),
    )


def main():
    args = parse_args()
    if args.image_size <= 0:
        args.image_size = clip_input_size(args.arch)

    device = resolve_device(args.device)
    use_amp = device.startswith("cuda")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = ImagePathDataset(
        img_dir=args.img_dir,
        output_root=args.output_root,
        image_size=args.image_size,
        include_path_contains=args.path_contains,
        skip_existing=args.skip_existing and not args.overwrite,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    encoder = CLIPPenultimateEncoder(args.arch).to(device)
    encoder.eval()

    raw_feature_dim = None
    raw_grid_size = None
    token_map_channels = normalize_optional_size(args.token_map_channels)
    token_map_grid = normalize_optional_size(args.token_map_grid)
    if args.latent_kind == "token_map":
        raw_feature_dim, raw_grid_size = encoder.infer_token_map_shape()
        feature_dim, grid_size = encoder.infer_token_map_shape(
            compress_channels=token_map_channels,
            compress_grid=token_map_grid,
        )
        latent_shape = (feature_dim, grid_size, grid_size)
    else:
        feature_dim = encoder.infer_feature_dim()
        grid_size = 1
        latent_shape = (feature_dim,)
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    meta_path = output_root / ".latent_cache_meta.pt"
    atomic_torch_save(
        {
            "arch": args.arch,
            "image_size": args.image_size,
            "feature_dim": feature_dim,
            "grid_size": grid_size,
            "latent_kind": args.latent_kind,
            "latent_shape": latent_shape,
            "save_dtype": args.save_dtype,
            "img_dir": str(Path(args.img_dir).resolve()),
            "source_feature_dim": raw_feature_dim if raw_feature_dim is not None else feature_dim,
            "source_grid_size": raw_grid_size if raw_grid_size is not None else grid_size,
            "token_map_channels": token_map_channels,
            "token_map_grid": token_map_grid,
            "compressed_token_map": args.latent_kind == "token_map",
        },
        meta_path,
    )

    print(f"[LatentCache] img_dir={Path(args.img_dir).resolve()}")
    print(f"[LatentCache] output_root={output_root}")
    print(f"[LatentCache] arch={args.arch}")
    print(f"[LatentCache] image_size={args.image_size}")
    print(f"[LatentCache] latent_kind={args.latent_kind}")
    if args.latent_kind == "token_map":
        print(f"[LatentCache] source_feature_dim={raw_feature_dim}")
        print(f"[LatentCache] source_grid_size={raw_grid_size}")
        print(f"[LatentCache] token_map_channels={token_map_channels}")
        print(f"[LatentCache] token_map_grid={token_map_grid}")
    print(f"[LatentCache] feature_dim={feature_dim}")
    print(f"[LatentCache] grid_size={grid_size}")
    print(f"[LatentCache] save_dtype={args.save_dtype}")
    print(f"[LatentCache] dataset_size={len(dataset)}")

    with torch.no_grad():
        pbar = tqdm(loader, desc="Encoding latents", mininterval=5.0)
        for x, rel_paths in pbar:
            x = x.to(device, non_blocking=True)
            x_clip = normalize_for_clip(x)
            with torch.cuda.amp.autocast(enabled=use_amp):
                latents = encode_batch(encoder, x_clip, args=args).to(dtype=save_dtype)

            for latent, rel_str in zip(latents, rel_paths):
                rel = Path(rel_str)
                out_path = latent_path_for_relative(output_root, rel)
                if out_path.exists() and not args.overwrite:
                    continue
                atomic_torch_save(latent.cpu().contiguous(), out_path)

    print(f"Latent cache complete: {output_root}")


if __name__ == "__main__":
    main()
