import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.latent_rectifier import build_latent_rectifier_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute delta latent cache from mirrored clean/SR latent trees")
    parser.add_argument("--clean_latent_root", type=str, required=True, help="Root directory with clean/original latent .pt files")
    parser.add_argument("--sr_latent_root", type=str, required=True, help="Root directory with SR latent .pt files")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory to store delta latent .pt files")
    parser.add_argument("--rectifier_ckpt", type=str, required=True, help="Path to trained latent rectifier checkpoint")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples whose delta latent already exists")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing delta latent files")
    parser.add_argument(
        "--delta_mode",
        type=str,
        choices=["orig_minus_rectified", "sr_minus_rectified"],
        default="orig_minus_rectified",
        help="How to derive delta from clean/sr/rectified latents",
    )
    parser.add_argument(
        "--path_contains",
        type=str,
        nargs="+",
        default=None,
        help="Only include latent files whose relative path contains one of these substrings.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def latent_path_for_relative(output_root: Path, rel: Path) -> Path:
    return (output_root / rel).with_suffix(".pt")


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class PairedLatentPathDataset(Dataset):
    def __init__(
        self,
        clean_latent_root: str,
        sr_latent_root: str,
        output_root: str,
        include_path_contains=None,
        exclude_hidden: bool = True,
        skip_existing: bool = False,
    ):
        self.clean_root = Path(clean_latent_root).resolve()
        self.sr_root = Path(sr_latent_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.include_path_contains = tuple(include_path_contains or [])
        self.exclude_hidden = exclude_hidden
        self.clean_meta = self._load_meta(self.clean_root)
        self.sr_meta = self._load_meta(self.sr_root)

        clean_paths = sorted(path for path in self.clean_root.rglob("*.pt") if self._is_valid_latent_path(path))
        self.total_pair_count = 0
        self.probe_clean_path = None
        self.samples = []
        missing = []
        for clean_path in clean_paths:
            rel = clean_path.relative_to(self.clean_root)
            sr_path = self.sr_root / rel
            if not sr_path.is_file():
                missing.append(str(rel))
                continue
            self.total_pair_count += 1
            if self.probe_clean_path is None:
                self.probe_clean_path = clean_path
            out_path = latent_path_for_relative(self.output_root, rel)
            if skip_existing and out_path.is_file():
                continue
            self.samples.append((clean_path, sr_path, rel))

        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"SR latent cache not found for {len(missing)} files under {self.sr_root}.\nExamples:\n{preview}"
            )

        if self.total_pair_count == 0:
            raise ValueError(f"No mirrored clean/SR latent pairs found under {self.clean_root} and {self.sr_root}")

        sample = self._load_latent(self.probe_clean_path)
        self.sample_shape = tuple(sample.shape)
        self.feature_dim = int(sample.shape[0])
        self.grid_size = int(sample.shape[-1]) if sample.ndim == 3 else 1
        self.latent_kind = self.clean_meta.get("latent_kind") or self.sr_meta.get("latent_kind") or ("token_map" if sample.ndim == 3 else "cls")
        self.clean_img_dir = self.clean_meta.get("img_dir")
        self.sr_img_dir = self.sr_meta.get("img_dir")

        clean_kind = self.clean_meta.get("latent_kind")
        sr_kind = self.sr_meta.get("latent_kind")
        if clean_kind and sr_kind and clean_kind != sr_kind:
            raise ValueError(f"Latent kind mismatch: clean={clean_kind}, sr={sr_kind}")

        clean_feature_dim = self.clean_meta.get("feature_dim")
        sr_feature_dim = self.sr_meta.get("feature_dim")
        if clean_feature_dim and sr_feature_dim and int(clean_feature_dim) != int(sr_feature_dim):
            raise ValueError(f"Feature dim mismatch: clean={clean_feature_dim}, sr={sr_feature_dim}")

        clean_grid = self.clean_meta.get("grid_size")
        sr_grid = self.sr_meta.get("grid_size")
        if clean_grid and sr_grid and int(clean_grid) != int(sr_grid):
            raise ValueError(f"Grid size mismatch: clean={clean_grid}, sr={sr_grid}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_meta(root: Path):
        meta_path = root / ".latent_cache_meta.pt"
        if meta_path.is_file():
            return torch.load(meta_path, map_location="cpu")
        return {}

    def _is_hidden_path(self, path: Path) -> bool:
        rel = path.relative_to(self.clean_root)
        return any(part.startswith(".") for part in rel.parts)

    def _matches_include_filter(self, path: Path) -> bool:
        if not self.include_path_contains:
            return True
        path_str = str(path)
        return any(token in path_str for token in self.include_path_contains)

    def _is_valid_latent_path(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if path.suffix.lower() != ".pt":
            return False
        if path.name.startswith("."):
            return False
        if self.exclude_hidden and self._is_hidden_path(path):
            return False
        return self._matches_include_filter(path)

    @staticmethod
    def _extract_latent(obj):
        if torch.is_tensor(obj):
            latent = obj
        elif isinstance(obj, dict):
            if "latent" in obj:
                latent = obj["latent"]
            elif "feature" in obj:
                latent = obj["feature"]
            elif "tensor" in obj:
                latent = obj["tensor"]
            else:
                raise KeyError("Latent checkpoint dict must contain one of: latent, feature, tensor")
        else:
            raise TypeError(f"Unsupported latent object type: {type(obj)}")

        latent = latent.detach().to(dtype=torch.float32)
        if latent.ndim not in (1, 3):
            raise ValueError(f"Expected latent vector (C,) or map (C,H,W), got shape={tuple(latent.shape)}")
        return latent.contiguous()

    def _load_latent(self, path: Path):
        obj = torch.load(path, map_location="cpu")
        return self._extract_latent(obj)

    def __getitem__(self, idx):
        clean_path, sr_path, rel = self.samples[idx]
        z_clean = self._load_latent(clean_path)
        z_sr = self._load_latent(sr_path)
        return z_clean, z_sr, str(rel)


def build_meta(args, dataset: PairedLatentPathDataset):
    return {
        "feature_dim": dataset.feature_dim,
        "grid_size": dataset.grid_size,
        "latent_kind": dataset.latent_kind,
        "latent_shape": dataset.sample_shape,
        "save_dtype": args.save_dtype,
        "clean_latent_root": str(Path(args.clean_latent_root).resolve()),
        "sr_latent_root": str(Path(args.sr_latent_root).resolve()),
        "clean_img_dir": dataset.clean_img_dir,
        "sr_img_dir": dataset.sr_img_dir,
        "output_root": str(Path(args.output_root).resolve()),
        "delta_mode": args.delta_mode,
        "rectifier_ckpt": str(Path(args.rectifier_ckpt).resolve()),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = device.startswith("cuda")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = PairedLatentPathDataset(
        clean_latent_root=args.clean_latent_root,
        sr_latent_root=args.sr_latent_root,
        output_root=args.output_root,
        include_path_contains=args.path_contains,
        skip_existing=args.skip_existing and not args.overwrite,
    )

    meta_path = output_root / ".latent_cache_meta.pt"
    atomic_torch_save(build_meta(args, dataset), meta_path)

    print(f"[DeltaLatent] clean_latent_root={Path(args.clean_latent_root).resolve()}")
    print(f"[DeltaLatent] sr_latent_root={Path(args.sr_latent_root).resolve()}")
    print(f"[DeltaLatent] output_root={output_root}")
    print(f"[DeltaLatent] rectifier_ckpt={Path(args.rectifier_ckpt).resolve()}")
    print(f"[DeltaLatent] delta_mode={args.delta_mode}")
    print(f"[DeltaLatent] latent_kind={dataset.latent_kind}")
    print(f"[DeltaLatent] feature_dim={dataset.feature_dim}")
    print(f"[DeltaLatent] grid_size={dataset.grid_size}")
    print(f"[DeltaLatent] save_dtype={args.save_dtype}")
    print(f"[DeltaLatent] total_pair_count={dataset.total_pair_count}")
    print(f"[DeltaLatent] pending_count={len(dataset)}")

    if len(dataset) == 0:
        print(f"Delta cache already complete: {output_root}")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    checkpoint = torch.load(args.rectifier_ckpt, map_location="cpu")
    rectifier, rectifier_meta = build_latent_rectifier_from_checkpoint(
        checkpoint,
        input_dim=dataset.feature_dim,
    )
    rectifier = rectifier.to(device)
    rectifier.eval()
    print(
        f"[DeltaLatent] rectifier_kind={rectifier_meta['rectifier_kind']} "
        f"hidden_dim={rectifier_meta['hidden_dim']} depth={rectifier_meta['depth']}"
    )

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    with torch.no_grad():
        pbar = tqdm(loader, desc="Encoding delta latents", mininterval=5.0)
        for z_clean, z_sr, rel_paths in pbar:
            z_clean = z_clean.to(device, non_blocking=True)
            z_sr = z_sr.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z_hat = rectifier(z_sr).float()
                if args.delta_mode == "sr_minus_rectified":
                    delta = z_sr.float() - z_hat
                else:
                    delta = z_clean.float() - z_hat
                delta = delta.to(dtype=save_dtype)

            for delta_item, rel_str in zip(delta, rel_paths):
                rel = Path(rel_str)
                out_path = latent_path_for_relative(output_root, rel)
                if out_path.exists() and not args.overwrite:
                    continue
                atomic_torch_save(delta_item.cpu().contiguous(), out_path)

    print(f"Delta latent cache complete: {output_root}")


if __name__ == "__main__":
    main()
