import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SRRectifyDataset(Dataset):
    """
    Returns only the original real image x in [0,1].
    SR(D(x)) is produced inside the training loop with a frozen SR model.
    """

    def __init__(self, img_dir: str, image_size: int = 256, sr_cache_root: str = None):
        self.img_root = Path(img_dir).resolve()
        self.sr_cache_root = Path(sr_cache_root).resolve() if sr_cache_root else None
        self.paths = sorted(
            p for p in self.img_root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        if not self.paths:
            raise ValueError(f"No images found in: {img_dir}")

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def _resolve_cache_path(self, img_path: Path):
        if self.sr_cache_root is None:
            return None
        rel = img_path.relative_to(self.img_root)
        candidates = [self.sr_cache_root / rel, self.sr_cache_root / self.img_root.name / rel]
        for direct in candidates:
            if direct.exists():
                return direct
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                cand = direct.with_suffix(ext)
                if cand.exists():
                    return cand
        return None

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        if self.sr_cache_root is None:
            return x

        cache_path = self._resolve_cache_path(img_path)
        if cache_path is None:
            raise FileNotFoundError(
                f"SR cache not found for {img_path}. "
                f"Expected under {self.sr_cache_root}"
            )
        x_sr = self.transform(Image.open(cache_path).convert("RGB"))
        return x, x_sr
