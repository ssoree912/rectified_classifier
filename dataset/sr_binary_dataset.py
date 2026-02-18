from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class SRBinaryDataset(Dataset):
    """
    Folder structure:
      root/real/... and root/fake/...
    Also supports:
      root/nature/... and root/ai/...
    Also supports:
      root/real/... and root/<multiple fake-generator dirs>/...
    """

    def __init__(self, root, sr_cache_root, image_size=256):
        self.root = Path(root).resolve()
        self.sr_cache_root = Path(sr_cache_root).resolve()
        self.tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

        real_dir = self.root / "real"
        fake_dir = self.root / "fake"
        if not real_dir.is_dir():
            real_dir = self.root / "nature"
        if not fake_dir.is_dir():
            fake_dir = self.root / "ai"

        fake_dirs = []
        if real_dir.is_dir() and fake_dir.is_dir():
            fake_dirs = [fake_dir]
        elif real_dir.is_dir():
            for d in sorted(self.root.iterdir()):
                if not d.is_dir():
                    continue
                if d == real_dir:
                    continue
                if d.name.startswith("."):
                    continue
                fake_dirs.append(d)

        if not real_dir.is_dir() or not fake_dirs:
            raise ValueError(
                f"Could not find valid class folders under {self.root}. "
                "Expected (real,fake), (nature,ai), or (real + multiple fake folders)."
            )

        self.items = []
        for p in real_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                self.items.append((p, 0))
        for d in fake_dirs:
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    self.items.append((p, 1))
        self.items.sort(key=lambda x: str(x[0]))
        if not self.items:
            raise ValueError(f"No images found under: {self.root}")

    def _sr_path(self, x_path: Path):
        rel = x_path.relative_to(self.root)
        candidates = [
            self.sr_cache_root / rel,
            self.sr_cache_root / self.root.name / rel,
        ]
        for p in candidates:
            if p.exists():
                return p
            for ext in VALID_EXTS:
                p2 = p.with_suffix(ext)
                if p2.exists():
                    return p2
        return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x_path, y = self.items[idx]
        sr_path = self._sr_path(x_path)
        if sr_path is None:
            raise FileNotFoundError(f"SR cache missing for: {x_path}")

        x = self.tf(Image.open(x_path).convert("RGB"))
        x_sr = self.tf(Image.open(sr_path).convert("RGB"))
        return x, x_sr, torch.tensor(y, dtype=torch.long)
