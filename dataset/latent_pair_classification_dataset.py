from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentPairClassificationDataset(Dataset):
    def __init__(
        self,
        clean_latent_root: str,
        aux_latent_root: str,
        split: str,
        include_path_contains=None,
        exclude_hidden: bool = True,
    ):
        self.clean_root = Path(clean_latent_root).resolve() / split
        self.aux_root = Path(aux_latent_root).resolve() / split
        self.include_path_contains = tuple(include_path_contains or [])
        self.exclude_hidden = exclude_hidden
        self.clean_meta = self._load_meta(self.clean_root.parent)
        self.aux_meta = self._load_meta(self.aux_root.parent)

        if not self.clean_root.is_dir():
            raise FileNotFoundError(f"Clean latent split not found: {self.clean_root}")
        if not self.aux_root.is_dir():
            raise FileNotFoundError(f"Aux latent split not found: {self.aux_root}")

        clean_paths = sorted(path for path in self.clean_root.rglob("*.pt") if self._is_valid_latent_path(path))
        if not clean_paths:
            raise ValueError(f"No latent files found under: {self.clean_root}")

        self.samples = []
        missing = []
        label_counts = {0: 0, 1: 0}
        for clean_path in clean_paths:
            rel = clean_path.relative_to(self.clean_root)
            aux_path = self.aux_root / rel
            if not aux_path.is_file():
                missing.append(str(rel))
                continue
            label = self._label_from_relative(rel)
            label_counts[label] += 1
            self.samples.append((clean_path, aux_path, label, rel))

        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"Aux latent cache not found for {len(missing)} files under {self.aux_root}.\nExamples:\n{preview}"
            )

        sample_z = self._load_latent(self.samples[0][0])
        self.feature_dim = int(sample_z.shape[0])
        self.latent_ndim = int(sample_z.ndim)
        self.is_spatial = self.latent_ndim == 3
        self.grid_size = int(sample_z.shape[-1]) if self.is_spatial else 1
        self.clean_latent_kind = self.clean_meta.get("latent_kind") or ("token_map" if self.is_spatial else "cls")
        self.aux_latent_kind = self.aux_meta.get("latent_kind") or self.clean_latent_kind
        self.class_counts = label_counts

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

    @staticmethod
    def _label_from_relative(rel: Path) -> int:
        for part in rel.parts:
            if part == "0_real":
                return 0
            if part == "1_fake":
                return 1
        raise ValueError(f"Unable to infer binary label from latent path: {rel}")

    def __getitem__(self, idx):
        clean_path, aux_path, label, rel = self.samples[idx]
        z_clean = self._load_latent(clean_path)
        z_aux = self._load_latent(aux_path)
        return z_clean, z_aux, torch.tensor(label, dtype=torch.float32), str(rel)
