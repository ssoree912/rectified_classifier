from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class SRRectifyDataset(Dataset):
    """
    Returns paired tensors (x, x_sr) where x is the original image and
    x_sr is the cached SR image at the same relative path under sr_cache_root.
    """

    def __init__(
        self,
        img_dir: str,
        image_size: int = 256,
        sr_cache_root: str = None,
        include_path_contains=None,
        exclude_hidden: bool = True,
    ):
        self.img_root = Path(img_dir).resolve()
        self.sr_cache_root = Path(sr_cache_root).resolve() if sr_cache_root else None
        self.include_path_contains = tuple(include_path_contains or [])
        self.exclude_hidden = exclude_hidden

        image_paths = sorted(p for p in self.img_root.rglob("*") if self._is_valid_image_path(p))
        if not image_paths:
            raise ValueError(f"No images found in: {img_dir}")

        if self.sr_cache_root is None:
            self.samples = image_paths
        else:
            self.samples = []
            missing = []
            for img_path in image_paths:
                rel = img_path.relative_to(self.img_root)
                cache_path = self.sr_cache_root / rel
                if not cache_path.is_file():
                    missing.append(str(rel))
                    continue
                self.samples.append((img_path, cache_path))

            if missing:
                preview = "\n".join(missing[:5])
                raise FileNotFoundError(
                    f"SR cache not found for {len(missing)} images under {self.sr_cache_root}.\n"
                    f"Examples:\n{preview}"
                )

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def _is_hidden_path(self, img_path: Path) -> bool:
        rel = img_path.relative_to(self.img_root)
        return any(part.startswith(".") for part in rel.parts)

    def _matches_include_filter(self, img_path: Path) -> bool:
        if not self.include_path_contains:
            return True
        img_path_str = str(img_path)
        return any(token in img_path_str for token in self.include_path_contains)

    def _is_valid_image_path(self, img_path: Path) -> bool:
        if not img_path.is_file():
            return False
        if img_path.suffix.lower() not in VALID_EXTS:
            return False
        if self.exclude_hidden and self._is_hidden_path(img_path):
            return False
        return self._matches_include_filter(img_path)

    def __getitem__(self, idx):
        if self.sr_cache_root is None:
            img_path = self.samples[idx]
            with Image.open(img_path) as img:
                x = self.transform(img.convert("RGB"))
            return x

        img_path, cache_path = self.samples[idx]
        with Image.open(img_path) as img:
            x = self.transform(img.convert("RGB"))
        with Image.open(cache_path) as sr_img:
            x_sr = self.transform(sr_img.convert("RGB"))
        return x, x_sr
