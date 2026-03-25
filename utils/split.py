from pathlib import Path


def check_dataset(data_root: str):
    root = Path(data_root)
    for split in ["train", "val", "test"]:
        image_dir = root / split
        mask_dir = root / f"{split}_labels"
        image_names = sorted([p.stem for p in image_dir.glob("*.tiff")])
        mask_names = sorted([p.stem for p in mask_dir.glob("*.tif")])
        matched = sorted(set(image_names) & set(mask_names))
        print(f"{split}: images={len(image_names)}, masks={len(mask_names)}, matched={len(matched)}")


if __name__ == "__main__":
    check_dataset("data/tiff")
