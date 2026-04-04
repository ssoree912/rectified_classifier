import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_split_manifest(root: str, split: str):
    root_path = Path(root).resolve()
    manifest_path = root_path / split / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Memmap manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["_root"] = root_path
    manifest["_split_root"] = root_path / split
    manifest["_manifest_path"] = manifest_path
    return manifest


def resolve_shard_file(manifest: dict, shard: dict, key: str) -> Path:
    filename = shard.get(f"{key}_file")
    if not filename:
        raise KeyError(f"Shard is missing file entry for key={key}")
    return manifest["_split_root"] / filename


class LatentMemmapPairDataset(Dataset):
    def __init__(
        self,
        clean_root: str,
        aux_root: str,
        split: str,
        aux_key: str,
        include_path_contains=None,
        return_label: bool = True,
        return_relpath: bool = False,
    ):
        self.clean_manifest = load_split_manifest(clean_root, split)
        self.aux_manifest = load_split_manifest(aux_root, split)
        self.split = split
        self.aux_key = str(aux_key)
        self.include_path_contains = tuple(include_path_contains or [])
        self.return_label = bool(return_label)
        self.return_relpath = bool(return_relpath)

        self._validate_manifests()

        self.feature_dim = int(self.clean_manifest["feature_dim"])
        self.grid_size = int(self.clean_manifest.get("grid_size", 1))
        self.latent_kind = str(self.clean_manifest.get("latent_kind", "cls"))
        self.is_spatial = bool(self.grid_size > 1)
        self.class_counts = self._compute_class_counts()

        self._clean_arrays = {}
        self._aux_arrays = {}
        self._label_arrays = {}
        self._path_lists = {}

        self._shard_counts = [int(shard["count"]) for shard in self.clean_manifest["shards"]]
        self._prefix_counts = []
        total = 0
        for count in self._shard_counts:
            total += count
            self._prefix_counts.append(total)

        self._filtered_indices = None
        if self.include_path_contains:
            self._filtered_indices = self._build_filtered_indices()

    def __len__(self):
        if self._filtered_indices is not None:
            return len(self._filtered_indices)
        return int(self.clean_manifest["total_count"])

    def _validate_manifests(self):
        clean_shards = self.clean_manifest["shards"]
        aux_shards = self.aux_manifest["shards"]
        if len(clean_shards) != len(aux_shards):
            raise ValueError(
                f"Shard count mismatch between clean and aux manifests: {len(clean_shards)} vs {len(aux_shards)}"
            )

        clean_shape = tuple(self.clean_manifest["latent_shape"])
        aux_shape = tuple(self.aux_manifest["latent_shape"])
        if clean_shape != aux_shape:
            raise ValueError(f"Latent shape mismatch: clean={clean_shape}, aux={aux_shape}")

        for idx, (clean_shard, aux_shard) in enumerate(zip(clean_shards, aux_shards)):
            clean_count = int(clean_shard["count"])
            aux_count = int(aux_shard["count"])
            if clean_count != aux_count:
                raise ValueError(f"Shard size mismatch at shard {idx}: clean={clean_count}, aux={aux_count}")
            if f"{self.aux_key}_file" not in aux_shard:
                raise KeyError(f"Aux manifest shard {idx} is missing key={self.aux_key}")
            if "clean_file" not in clean_shard:
                raise KeyError(f"Clean manifest shard {idx} is missing key=clean")
            if "label_file" not in clean_shard or "path_file" not in clean_shard:
                raise KeyError(f"Clean manifest shard {idx} must contain label_file and path_file")

    def _compute_class_counts(self):
        counts = {0: 0, 1: 0}
        for shard in self.clean_manifest["shards"]:
            labels = np.load(resolve_shard_file(self.clean_manifest, shard, "label"), mmap_mode="r")
            labels = np.asarray(labels, dtype=np.int64)
            counts[0] += int((labels == 0).sum())
            counts[1] += int((labels == 1).sum())
        return counts

    def _load_path_list(self, shard_idx: int):
        if shard_idx not in self._path_lists:
            shard = self.clean_manifest["shards"][shard_idx]
            path_file = resolve_shard_file(self.clean_manifest, shard, "path")
            with path_file.open("r", encoding="utf-8") as handle:
                self._path_lists[shard_idx] = [line.rstrip("\n") for line in handle]
        return self._path_lists[shard_idx]

    def _build_filtered_indices(self):
        indices = []
        for shard_idx, _shard in enumerate(self.clean_manifest["shards"]):
            relpaths = self._load_path_list(shard_idx)
            for local_idx, relpath in enumerate(relpaths):
                if any(token in relpath for token in self.include_path_contains):
                    indices.append((shard_idx, local_idx))
        return indices

    def _resolve_index(self, idx: int):
        if self._filtered_indices is not None:
            return self._filtered_indices[idx]
        shard_idx = bisect.bisect_right(self._prefix_counts, idx)
        prev_total = 0 if shard_idx == 0 else self._prefix_counts[shard_idx - 1]
        local_idx = idx - prev_total
        return shard_idx, local_idx

    def _load_array(self, cache: dict, manifest: dict, shard_idx: int, key: str):
        if shard_idx not in cache:
            path = resolve_shard_file(manifest, manifest["shards"][shard_idx], key)
            cache[shard_idx] = np.load(path, mmap_mode="r")
        return cache[shard_idx]

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._resolve_index(idx)

        clean_arr = self._load_array(self._clean_arrays, self.clean_manifest, shard_idx, "clean")
        aux_arr = self._load_array(self._aux_arrays, self.aux_manifest, shard_idx, self.aux_key)
        z_clean = torch.from_numpy(np.array(clean_arr[local_idx], copy=True)).float()
        z_aux = torch.from_numpy(np.array(aux_arr[local_idx], copy=True)).float()

        items = [z_clean, z_aux]
        if self.return_label:
            label_arr = self._load_array(self._label_arrays, self.clean_manifest, shard_idx, "label")
            label = float(label_arr[local_idx])
            items.append(torch.tensor(label, dtype=torch.float32))
        if self.return_relpath:
            relpath = self._load_path_list(shard_idx)[local_idx]
            items.append(relpath)
        return tuple(items)
