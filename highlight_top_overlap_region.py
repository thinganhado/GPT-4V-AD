#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries


def parse_color(color_str: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("--color must be R,G,B")
    vals = tuple(int(x) for x in parts)
    for v in vals:
        if v < 0 or v > 255:
            raise ValueError("Color channels must be in [0,255]")
    return vals


def pick_top_regions(csv_path: Path, method_filter: Optional[str]) -> Dict[str, Dict[str, int]]:
    best: Dict[str, Dict[str, int]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = row.get("image", "").strip()
            method = row.get("method", "").strip()
            if not image:
                continue
            if method_filter and method != method_filter:
                continue
            try:
                rid = int(row["region_id"])
                overlap = int(float(row["overlap_pixels"]))
            except Exception:
                continue
            cur = best.get(image)
            if cur is None or overlap > cur["overlap_pixels"]:
                best[image] = {
                    "region_id": rid,
                    "overlap_pixels": overlap,
                    "method": method,
                }
    return best


def load_region_mask(masks_path: Path, region_id: int) -> np.ndarray:
    obj = torch.load(masks_path, map_location="cpu", weights_only=False)
    masks = obj.get("masks", None) if isinstance(obj, dict) else None
    if not isinstance(masks, (list, tuple)) or len(masks) == 0:
        raise ValueError(f"Invalid masks file: {masks_path}")
    idx = region_id - 1
    if idx < 0 or idx >= len(masks):
        raise IndexError(f"region_id={region_id} out of range for {masks_path.name}; max={len(masks)}")
    return np.asarray(masks[idx]).astype(bool)


def draw_boundary_on_image(
    rgb: np.ndarray,
    region_mask: np.ndarray,
    color_rgb: Tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    boundary = find_boundaries(region_mask, mode="inner")
    if thickness > 1:
        boundary = binary_dilation(boundary, iterations=thickness - 1)
    boundary_u8 = (boundary.astype(np.uint8) * 255)

    if boundary_u8.shape[:2] != rgb.shape[:2]:
        boundary_u8 = cv2.resize(boundary_u8, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    boundary_bool = boundary_u8 > 0

    out = rgb.copy()
    out[boundary_bool] = np.array(color_rgb, dtype=np.uint8)
    return out


def draw_grid_and_ids(rgb: np.ndarray, div_num: int = 4) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    cell_w = w / float(div_num)
    cell_h = h / float(div_num)

    # White grid lines.
    for i in range(1, div_num):
        x = int(round(i * cell_w))
        y = int(round(i * cell_h))
        cv2.line(out, (x, 0), (x, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(out, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Highlight top-overlap region per sample on fake_spec_with_mask images."
    )
    parser.add_argument(
        "--stats-csv",
        required=True,
        help="Path to region_diff_stats.csv",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing fake_spec_with_mask__<sample>.png",
    )
    parser.add_argument(
        "--masks-root",
        required=True,
        help="Root that contains per-method masks, e.g. .../img/specs",
    )
    parser.add_argument(
        "--method",
        default="grid",
        help="Method filter used in CSV and masks path subfolder.",
    )
    parser.add_argument(
        "--image-pattern",
        default="fake_spec_with_mask__{sample}.png",
        help="Image filename pattern.",
    )
    parser.add_argument(
        "--sample-id",
        default="",
        help="Optional single sample_id to process.",
    )
    parser.add_argument(
        "--draw-grid",
        action="store_true",
        default=True,
        help="Overlay 4x4 grid lines and region numbers (grid-style).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to save highlighted images.",
    )
    parser.add_argument("--thickness", type=int, default=5, help="Boundary thickness in pixels.")
    parser.add_argument("--color", default="255,0,0", help="Boundary RGB color.")
    args = parser.parse_args()

    stats_csv = Path(args.stats_csv).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    masks_root = Path(args.masks_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    color_rgb = parse_color(args.color)
    tops = pick_top_regions(stats_csv, args.method if args.method else None)
    if args.sample_id:
        tops = {k: v for k, v in tops.items() if k == args.sample_id}

    if not tops:
        print("[WARN] No matching rows found in stats CSV.")
        return

    for sample, info in tops.items():
        region_id = int(info["region_id"])
        method = info["method"] if info["method"] else args.method

        image_path = images_dir / args.image_pattern.format(sample=sample)
        if not image_path.exists():
            print(f"[WARN] Missing image for {sample}: {image_path}")
            continue

        masks_path = masks_root / method / f"{sample}_{method}_masks.pth"
        if not masks_path.exists():
            print(f"[WARN] Missing masks for {sample}: {masks_path}")
            continue

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if args.draw_grid:
            rgb = draw_grid_and_ids(rgb, div_num=4)

        mask = load_region_mask(masks_path, region_id)
        highlighted = draw_boundary_on_image(rgb, mask, color_rgb, args.thickness)

        out_path = out_dir / f"{sample}_top_overlap_region{region_id}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR))
        print(f"[OK] {sample} region={region_id} overlap={info['overlap_pixels']} -> {out_path}")


if __name__ == "__main__":
    main()
