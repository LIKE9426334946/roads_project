from typing import Callable, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
import torchvision.transforms as T


class SegCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomScale:
    def __init__(self, scale_range=(0.75, 1.25)):
        self.scale_range = scale_range

    def __call__(self, image: Image.Image, mask: Image.Image):
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        w, h = image.size
        new_size = (max(int(h * scale), 32), max(int(w * scale), 32))
        image = TF.resize(image, new_size, interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, new_size, interpolation=T.InterpolationMode.NEAREST)
        return image, mask


class RandomCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image, mask: Image.Image):
        crop_h, crop_w = self.size
        w, h = image.size

        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)
            w, h = image.size

        i, j, hh, ww = T.RandomCrop.get_params(image, output_size=(crop_h, crop_w))
        image = TF.crop(image, i, j, hh, ww)
        mask = TF.crop(mask, i, j, hh, ww)
        return image, mask


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if np.random.rand() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if np.random.rand() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


class RandomRotate90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if np.random.rand() < self.p:
            k = int(np.random.choice([1, 2, 3]))
            image = TF.rotate(image, 90 * k)
            mask = TF.rotate(mask, 90 * k)
        return image, mask


class ColorJitterOnlyImage:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image: Image.Image, mask: Image.Image):
        return self.jitter(image), mask


class ToTensorAndNormalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        return image, mask


def get_transforms(crop_size=(512, 512), eval_size=(1024, 1024)) -> Dict[str, Callable]:
    train_tf = SegCompose(
        [
            RandomScale((0.75, 1.25)),
            RandomCrop(crop_size),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.2),
            RandomRotate90(0.5),
            ColorJitterOnlyImage(),
            ToTensorAndNormalize(),
        ]
    )
    eval_tf = SegCompose([Resize(eval_size), ToTensorAndNormalize()])
    return {"train": train_tf, "eval": eval_tf}
