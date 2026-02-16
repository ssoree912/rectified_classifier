import os
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class SRResidualRectifyDataset(Dataset):
    """
    Returns:
      r_clean: (3,H,W) = x_sr - x
      r_noisy: r_clean + Gaussian noise
    """

    def __init__(
        self,
        img_dir,
        sr_cache_root,
        image_size=256,
        noise_std=0.02,
        use_abs=False,
    ):
        self.img_dir = Path(img_dir).resolve()
        self.sr_cache_root = Path(sr_cache_root).resolve()
        self.noise_std = noise_std
        self.use_abs = use_abs

        self.paths = sorted(
            p for p in self.img_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        )
        if not self.paths:
            raise ValueError(f"No images found in: {self.img_dir}")

        self.tf = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def _sr_path(self, x_path: Path):
        rel = x_path.relative_to(self.img_dir)
        candidates = [
            self.sr_cache_root / rel,
            self.sr_cache_root / self.img_dir.name / rel,
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
        return len(self.paths)

    def __getitem__(self, idx):
        x_path = self.paths[idx]
        sr_path = self._sr_path(x_path)
        if sr_path is None:
            raise FileNotFoundError(f"SR cache missing for: {x_path}")

        x = self.tf(Image.open(x_path).convert("RGB"))
        x_sr = self.tf(Image.open(sr_path).convert("RGB"))

        r_clean = x_sr - x
        if self.use_abs:
            r_clean = r_clean.abs()

        if self.noise_std > 0:
            r_noisy = r_clean + torch.randn_like(r_clean) * self.noise_std
        else:
            r_noisy = r_clean.clone()
        return r_clean, r_noisy
