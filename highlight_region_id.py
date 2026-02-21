#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Highlight one region_id boundary from region_division.py masks with thick red lines."
    )
    parser.add_argument("--image", required=True, help="Path to source spectrogram PNG.")
    parser.add_argument("--masks-pth", required=True, help="Path to *_masks.pth created by region_division.py.")
    parser.add_argument("--region-id", required=True, type=int, help="1-based region id to highlight.")
    parser.add_argument("--output", default=None, help="Output PNG path. Default: <image>_region<id>_highlight.png")
    parser.add_argument(
        "--thickness",
        type=int,
        default=4,
        help="Boundary thickness in pixels (>=1).",
    )
    parser.add_argument(
        "--color",
        default="255,0,0",
        help="Boundary RGB color as R,G,B. Default: 255,0,0 (red).",
    )
    return parser.parse_args()


def parse_color(color_str: str):
    parts = [p.strip() for p in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("--color must be in R,G,B format, e.g. 255,0,0")
    vals = tuple(int(x) for x in parts)
    for v in vals:
        if v < 0 or v > 255:
            raise ValueError("Each color channel must be in [0, 255]")
    return vals


def load_masks(masks_path: Path):
    obj = torch.load(masks_path, map_location="cpu")
    if not isinstance(obj, dict) or "masks" not in obj:
        raise ValueError(f"Invalid masks file format: {masks_path}")
    masks = obj["masks"]
    if not isinstance(masks, (list, tuple)) or len(masks) == 0:
        raise ValueError(f"No masks found in: {masks_path}")
    return masks


def main():
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    masks_path = Path(args.masks_pth).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks file not found: {masks_path}")
    if args.region_id < 1:
        raise ValueError("--region-id must be >= 1")
    if args.thickness < 1:
        raise ValueError("--thickness must be >= 1")

    color_rgb = parse_color(args.color)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    masks = load_masks(masks_path)
    idx = args.region_id - 1
    if idx >= len(masks):
        raise IndexError(f"region-id={args.region_id} out of range. Available: 1..{len(masks)}")

    mask = np.asarray(masks[idx]).astype(bool)
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match image shape {rgb.shape[:2]}.\n"
            f"Use matching image and masks pair."
        )

    # Region boundary only, then dilate to make it bold/thick.
    boundary = find_boundaries(mask, mode="inner")
    if args.thickness > 1:
        boundary = binary_dilation(boundary, iterations=args.thickness - 1)

    out = rgb.copy()
    out[boundary] = np.array(color_rgb, dtype=np.uint8)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = image_path.with_name(f"{image_path.stem}_region{args.region_id}_highlight.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(out_path.as_posix())


if __name__ == "__main__":
    main()
