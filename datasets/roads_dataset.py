from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["background", "road"]
ID2COLOR = {
    0: (0, 0, 0),
    1: (255, 255, 255),
}


class MassachusettsRoadsDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split
        self.mask_dir = self.root / f"{split}_labels"
        self.transform = transform
        self.names = sorted([p.stem for p in self.image_dir.glob("*.tiff")])

        if len(self.names) == 0:
            raise RuntimeError(f"No .tiff images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _rgb_mask_to_binary(mask_rgb: np.ndarray) -> np.ndarray:
        return (np.all(mask_rgb == np.array([255, 255, 255], dtype=np.uint8), axis=-1)).astype(np.uint8)

    def __getitem__(self, idx: int):
        stem = self.names[idx]
        image_path = self.image_dir / f"{stem}.tiff"
        mask_path = self.mask_dir / f"{stem}.tif"

        image = Image.open(image_path).convert("RGB")
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        mask = Image.fromarray(self._rgb_mask_to_binary(mask_rgb))

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {
            "image": image,
            "mask": mask.long(),
            "name": stem,
        }
