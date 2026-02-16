#!/usr/bin/env python3
"""
Crop irregular regions to transparent PNGs from region table rows.

Expected table columns:
- sample_id
- method
- region_id

Example:
python utils/crop_regions_transparent.py \
  --table-csv /path/region_phone_table_top3_all.csv \
  --masks-root /path/Ms_region_outputs \
  --image-root /path/specs \
  --output-dir /path/region_crops
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

METHOD_DEFINITION_MAP = {
    "GRID": "The crop is taken from a fixed square cell in an NxN grid over the spectrogram.",
    "SUPERPIXEL": "The crop is taken from an irregular region formed by grouping nearby pixels with similar appearance.",
    "SAM": "The crop is taken from a region that follows the visible edges of the pattern as closely as possible.",
}

DEFAULT_TABLE_CSV = "/scratch3/che489/Ha/interspeech/datasets/region_phone_table_top3_all_with_ptype_feature.csv"
DEFAULT_MASKS_ROOT = "/scratch3/che489/Ha/interspeech/localization/Ms_region_outputs/"
DEFAULT_IMAGE_ROOT = "/scratch3/che489/Ha/interspeech/localization/specs/"
ALLOWED_METHODS = {"grid", "sam", "superpixel"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export transparent crops for (sample_id, method, region_id).")
    p.add_argument("--table-csv", type=str, default=DEFAULT_TABLE_CSV, help="CSV with sample_id,method,region_id columns.")
    p.add_argument("--masks-root", type=str, default=DEFAULT_MASKS_ROOT, help="Root folder with <method>/*_masks.pth.")
    p.add_argument(
        "--image-root",
        type=str,
        default=DEFAULT_IMAGE_ROOT,
        help="Root folder for source images (searched recursively by stem).",
    )
    p.add_argument("--image-suffix", type=str, default=".png", help="Source image suffix to index.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for PNG crops.")
    p.add_argument("--pad", type=int, default=8, help="Padding around mask bbox in pixels.")
    p.add_argument(
        "--min-side",
        type=int,
        default=512,
        help="Upscale crop so min(width,height) reaches at least this value.",
    )
    p.add_argument(
        "--label-prefix",
        type=str,
        default="R",
        help="Region label prefix (text becomes '<prefix><region_id>').",
    )
    p.add_argument("--no-label", action="store_true", default=False, help="Disable region number label.")
    p.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing crop files.")
    p.add_argument(
        "--update-qwen-prompts",
        action="store_true",
        default=True,
        help="Update Qwen3-VL prompt files with the active crop method wording (default: enabled).",
    )
    p.add_argument(
        "--no-update-qwen-prompts",
        dest="update_qwen_prompts",
        action="store_false",
        help="Disable Qwen3-VL prompt file updates.",
    )
    p.add_argument(
        "--qwen-system-file",
        type=str,
        default=r"C:\Users\donga\OneDrive\Documents\GitHub\Qwen3-VL\prompts\region_forensics_system.txt",
        help="Path to Qwen3-VL system prompt file.",
    )
    p.add_argument(
        "--qwen-user-file",
        type=str,
        default=r"C:\Users\donga\OneDrive\Documents\GitHub\Qwen3-VL\prompts\region_forensics_user.txt",
        help="Path to Qwen3-VL user prompt file.",
    )
    return p.parse_args()


def read_rows(table_csv: Path) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    with open(table_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "method", "region_id"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must include columns: {sorted(required)}")
        for r in reader:
            sample_id = str(r["sample_id"]).strip()
            method = str(r["method"]).strip()
            if method.lower() not in ALLOWED_METHODS:
                continue
            region_id = int(r["region_id"])
            rows.append((sample_id, method, region_id))
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for x in rows:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def build_image_index(image_root: Path, suffix: str) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in image_root.rglob(f"*{suffix}"):
        idx.setdefault(p.stem, p)
    return idx


def load_region_mask(masks_root: Path, sample_id: str, method: str, region_id: int) -> Optional[np.ndarray]:
    if method.lower() not in ALLOWED_METHODS:
        return None
    pth = masks_root / method / f"{sample_id}_{method}_masks.pth"
    if not pth.exists():
        return None
    blob = torch.load(pth, map_location="cpu")
    masks = blob.get("masks", []) if isinstance(blob, dict) else []
    idx = region_id - 1
    if idx < 0 or idx >= len(masks):
        return None
    m = masks[idx]
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim != 2:
        return None
    return m.astype(bool)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_region_label(img_rgba: Image.Image, region_id: int, prefix: str) -> None:
    draw = ImageDraw.Draw(img_rgba)
    text = f"{prefix}{region_id}"
    w, h = img_rgba.size
    font_size = max(28, int(min(w, h) * 0.16))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = max(8, int(w * 0.03))
    y = max(8, int(h * 0.03))
    draw.rectangle([x - 6, y - 4, x + tw + 6, y + th + 4], fill=(0, 0, 0, 150))
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)


def upscale_if_needed(img_rgba: Image.Image, min_side: int) -> Image.Image:
    w, h = img_rgba.size
    ms = min(w, h)
    if ms >= min_side or ms <= 0:
        return img_rgba
    scale = float(min_side) / float(ms)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img_rgba.resize((nw, nh), Image.LANCZOS)


def _update_qwen_prompts(system_file: Path, user_file: Path, methods: Sequence[str]) -> None:
    methods_norm = sorted({m.strip().upper() for m in methods if m.strip()})
    if not methods_norm:
        return

    if len(methods_norm) == 1 and methods_norm[0] in METHOD_DEFINITION_MAP:
        m = methods_norm[0]
        method_text = f"{m}. Method definition: {METHOD_DEFINITION_MAP[m]}"
    else:
        listed = " / ".join(methods_norm)
        method_text = f"{listed}. Method definition: method-specific crop definition from preprocessing."

    system_line = (
        f"- P2: cropped region from the fake spectrogram. The crop is produced by {method_text} "
        "This crop method does not affect the artificial audio generation process."
    )

    if system_file.exists():
        system_txt = system_file.read_text(encoding="utf-8")
        system_txt = re.sub(
            r"^- P2:.*$",
            system_line,
            system_txt,
            flags=re.MULTILINE,
        )
        system_file.write_text(system_txt, encoding="utf-8")

    if user_file.exists():
        user_txt = user_file.read_text(encoding="utf-8")
        user_txt = re.sub(
            r"^P2 is produced by .*?(?:\r?\n|$)",
            "",
            user_txt,
            flags=re.MULTILINE,
        )
        user_file.write_text(user_txt.strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    table_csv = Path(args.table_csv)
    masks_root = Path(args.masks_root)
    image_root = Path(args.image_root)
    out_root = Path(args.output_dir)

    rows = read_rows(table_csv)
    methods_from_rows = [m for _, m, _ in rows]
    image_idx = build_image_index(image_root, args.image_suffix)
    out_root.mkdir(parents=True, exist_ok=True)

    missing_mask = 0
    missing_image = 0
    written = 0

    for sample_id, method, region_id in rows:
        mask = load_region_mask(masks_root, sample_id, method, region_id)
        if mask is None:
            missing_mask += 1
            continue
        src = image_idx.get(sample_id)
        if src is None:
            missing_image += 1
            continue

        img = Image.open(src).convert("RGB")
        if img.size[::-1] != mask.shape:
            img = img.resize((mask.shape[1], mask.shape[0]), Image.BICUBIC)

        # Flip image and mask vertically so frequency runs bottom-to-top in outputs.
        rgb = np.flipud(np.array(img, dtype=np.uint8))
        mask = np.flipud(mask)
        alpha = (mask.astype(np.uint8) * 255)
        rgba = np.dstack([rgb, alpha])

        bb = mask_bbox(mask)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        x1 = max(0, x1 - args.pad)
        y1 = max(0, y1 - args.pad)
        x2 = min(mask.shape[1] - 1, x2 + args.pad)
        y2 = min(mask.shape[0] - 1, y2 + args.pad)

        crop = Image.fromarray(rgba, mode="RGBA").crop((x1, y1, x2 + 1, y2 + 1))
        crop = upscale_if_needed(crop, args.min_side)
        if not args.no_label:
            draw_region_label(crop, region_id=region_id, prefix=args.label_prefix)

        method_dir = out_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        out_path = method_dir / f"{sample_id}__r{region_id}.png"
        if out_path.exists() and not args.overwrite:
            continue
        crop.save(out_path)
        written += 1

    print(f"rows_total={len(rows)}")
    print(f"written={written}")
    print(f"missing_mask={missing_mask}")
    print(f"missing_image={missing_image}")
    print(f"output_dir={out_root}")
    if args.update_qwen_prompts:
        _update_qwen_prompts(
            system_file=Path(args.qwen_system_file),
            user_file=Path(args.qwen_user_file),
            methods=methods_from_rows,
        )
        print(f"updated_qwen_system_file={Path(args.qwen_system_file)}")
        print(f"updated_qwen_user_file={Path(args.qwen_user_file)}")


if __name__ == "__main__":
    main()
