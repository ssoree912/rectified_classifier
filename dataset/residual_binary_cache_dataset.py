from pathlib import Path

import torch
from torch.utils.data import Dataset


class ResidualBinaryCacheDataset(Dataset):
    """
    Folder structure:
      root/real/...*.pt and root/fake/...*.pt
    Also supports:
      root/nature/...*.pt and root/ai/...*.pt
    """

    def __init__(self, root):
        self.root = Path(root).resolve()

        real_dir = self.root / "real"
        fake_dir = self.root / "fake"
        if not real_dir.is_dir():
            real_dir = self.root / "nature"
        if not fake_dir.is_dir():
            fake_dir = self.root / "ai"
        if not real_dir.is_dir() or not fake_dir.is_dir():
            raise ValueError(
                f"Could not find class folders under {self.root}. "
                "Expected (real,fake) or (nature,ai)."
            )

        self.items = []
        for d, y in [(real_dir, 0), (fake_dir, 1)]:
            for p in d.rglob("*.pt"):
                if p.is_file():
                    self.items.append((p, y))
        self.items.sort(key=lambda x: str(x[0]))
        if not self.items:
            raise ValueError(f"No residual .pt files found under: {self.root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        # Try faster loading path on newer PyTorch; fallback for compatibility.
        try:
            r = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            try:
                r = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                r = torch.load(path, map_location="cpu")
        if isinstance(r, dict):
            if "residual" in r:
                r = r["residual"]
            else:
                raise ValueError(f"Unsupported residual cache dict format: {path}")

        if not torch.is_tensor(r):
            raise TypeError(f"Residual cache is not a tensor: {path}")

        if r.ndim == 4 and r.shape[0] == 1:
            r = r[0]
        if r.ndim != 3:
            raise ValueError(f"Residual tensor must be CHW, got shape={tuple(r.shape)} in {path}")

        return r.float(), torch.tensor(y, dtype=torch.long)
