import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml
from torchvision.transforms import functional as TF
import torchvision.transforms as T

from datasets.roads_dataset import ID2COLOR
from models.unet_resnet_attn import UNetResNet34Attn
from utils.visualize import overlay


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/exp/checkpoints/best.pth")
    parser.add_argument("--input", type=str, required=True, help="single .tiff path or a directory of .tiff images")
    parser.add_argument("--output_dir", type=str, default="runs/infer")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    return parser.parse_args()


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in ID2COLOR.items():
        color[mask == class_id] = np.array(rgb, dtype=np.uint8)
    return color


def normalize_tile(image_np: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(image_np)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, MEAN, STD)
    return tensor


def sliding_window_predict(model, image: Image.Image, device, tile_size: int = 512, stride: int = 256):
    image_np = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = image_np.shape[:2]
    num_classes = model.head.out_channels

    prob_map = np.zeros((num_classes, h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(h - tile_size, 0) + 1, stride))
    xs = list(range(0, max(w - tile_size, 0) + 1, stride))
    if len(ys) == 0 or ys[-1] != h - tile_size:
        ys.append(max(h - tile_size, 0))
    if len(xs) == 0 or xs[-1] != w - tile_size:
        xs.append(max(w - tile_size, 0))

    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = image_np[y:y + tile_size, x:x + tile_size]
                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                inp = normalize_tile(tile).unsqueeze(0).to(device)
                logits = model(inp)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                probs = probs[:, : min(tile_size, h - y), : min(tile_size, w - x)]
                prob_map[:, y:y + probs.shape[1], x:x + probs.shape[2]] += probs
                count_map[y:y + probs.shape[1], x:x + probs.shape[2]] += 1.0

    prob_map /= np.maximum(count_map[None, :, :], 1e-6)
    pred = np.argmax(prob_map, axis=0).astype(np.uint8)
    return pred


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet34Attn(
        num_classes=cfg["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        pretrained=False,
        use_scse=cfg["model"]["use_scse"],
        use_aspp=cfg["model"]["use_aspp"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(input_path.glob("*.tiff"))) if input_path.is_dir() else [input_path]

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        pred = sliding_window_predict(model, image, device, tile_size=args.tile_size, stride=args.stride)
        pred_color = mask_to_color(pred)
        image_np = np.array(image, dtype=np.uint8)
        over = overlay(image_np, pred_color)

        stem = image_path.stem
        Image.fromarray(pred).save(output_dir / f"{stem}_mask.png")
        Image.fromarray(pred_color).save(output_dir / f"{stem}_color.png")
        Image.fromarray(over).save(output_dir / f"{stem}_overlay.png")

    print(f"Inference outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
