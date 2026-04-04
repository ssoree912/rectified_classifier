from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentRectifyDataset(Dataset):
    """
    Returns paired latent tensors (z, z_sr) loaded from mirrored `.pt` trees.
    Latents may be either global vectors with shape (C,) or spatial maps with shape (C, H, W).
    """

    def __init__(
        self,
        clean_latent_root: str,
        sr_latent_root: str,
        include_path_contains=None,
        exclude_hidden: bool = True,
    ):
        self.clean_root = Path(clean_latent_root).resolve()
        self.sr_root = Path(sr_latent_root).resolve()
        self.include_path_contains = tuple(include_path_contains or [])
        self.exclude_hidden = exclude_hidden
        self.clean_meta = self._load_meta(self.clean_root)
        self.sr_meta = self._load_meta(self.sr_root)

        clean_paths = sorted(path for path in self.clean_root.rglob("*.pt") if self._is_valid_latent_path(path))
        if not clean_paths:
            raise ValueError(f"No latent files found in: {self.clean_root}")

        self.samples = []
        missing = []
        for clean_path in clean_paths:
            rel = clean_path.relative_to(self.clean_root)
            sr_path = self.sr_root / rel
            if not sr_path.is_file():
                missing.append(str(rel))
                continue
            self.samples.append((clean_path, sr_path))

        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"SR latent cache not found for {len(missing)} files under {self.sr_root}.\n"
                f"Examples:\n{preview}"
            )

        sample_z = self._load_latent(self.samples[0][0])
        self.feature_dim = int(sample_z.shape[0])
        self.latent_ndim = int(sample_z.ndim)
        self.is_spatial = self.latent_ndim == 3
        self.grid_size = int(sample_z.shape[-1]) if self.is_spatial else 1
        self.latent_kind = self._infer_latent_kind()

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_meta(root: Path):
        meta_path = root / ".latent_cache_meta.pt"
        if meta_path.is_file():
            return torch.load(meta_path, map_location="cpu")
        return {}

    def _infer_latent_kind(self) -> str:
        meta_kind = self.clean_meta.get("latent_kind") or self.sr_meta.get("latent_kind")
        if meta_kind:
            return str(meta_kind)
        return "token_map" if self.is_spatial else "cls"

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
        clean_path, sr_path = self.samples[idx]
        z = self._load_latent(clean_path)
        z_sr = self._load_latent(sr_path)
        return z, z_sr
