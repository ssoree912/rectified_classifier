import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SRRectifyDataset(Dataset):
    """
    Returns only the original real image x in [0,1].
    SR(D(x)) is produced inside the training loop with a frozen SR model.
    """

    def __init__(self, img_dir: str, image_size: int = 256):
        self.paths = sorted(
            os.path.join(img_dir, name)
            for name in os.listdir(img_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
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

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        return x
