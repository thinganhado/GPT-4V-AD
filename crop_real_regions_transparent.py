#!/usr/bin/env python3
"""
Create transparent region crops from REAL audio using fake-region masks.

- Region geometry comes from fake sample masks: <masks-root>/<method>/<fake_stem>_<method>_masks.pth
- Real side is resolved from pairs CSV (real_path, fake_path, split)
- Supports two real-image sources:
  1) --real-spec-root: precomputed real spectrogram PNGs (matched by real stem)
  2) fallback: on-the-fly spectrogram from real audio path
- Output layout mirrors fake crop layout: <output-dir>/<method>/<fake_stem>__r<region_id>.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crop real-audio regions using fake masks.")
    p.add_argument("--table-csv", type=str, required=True, help="CSV with sample_id,method,region_id columns.")
    p.add_argument("--pairs-csv", type=str, required=True, help="pairs_vocv4.csv path.")
    p.add_argument("--masks-root", type=str, required=True, help="Root with <method>/*_masks.pth")
    p.add_argument("--output-dir", type=str, required=True, help="Output root.")
    p.add_argument(
        "--real-spec-root",
        type=str,
        default=None,
        help="Optional root of precomputed real spectrogram PNGs (matched by real stem).",
    )
    p.add_argument("--real-spec-suffix", type=str, default=".png", help="Suffix for precomputed real spectrograms.")

    # make_specs_768.py compatible params (used only when --real-spec-root is not set)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)

    p.add_argument("--pad", type=int, default=8)
    p.add_argument("--min-side", type=int, default=512)
    p.add_argument("--label-prefix", type=str, default="R")
    p.add_argument("--no-label", action="store_true", default=False)
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def read_table_rows(path: Path) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        need = {"sample_id", "method", "region_id"}
        if r.fieldnames is None or not need.issubset(set(r.fieldnames)):
            raise ValueError(f"table CSV must contain {sorted(need)}")
        for row in r:
            rows.append((str(row["sample_id"]).strip(), str(row["method"]).strip(), int(row["region_id"])))
    seen = set()
    out = []
    for x in rows:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def read_pairs_map(pairs_csv: Path) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    with open(pairs_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        req = {"real_path", "fake_path"}
        if reader.fieldnames is None or not req.issubset(set(reader.fieldnames)):
            raise ValueError("pairs CSV must include real_path and fake_path columns")
        for row in reader:
            real_path = Path(str(row["real_path"]).strip())
            fake_path = Path(str(row["fake_path"]).strip())
            m[fake_path.stem] = real_path
    return m


def build_spec_index(spec_root: Path, suffix: str) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in spec_root.rglob(f"*{suffix}"):
        idx.setdefault(p.stem, p)
    return idx


def load_region_mask(masks_root: Path, sample_id: str, method: str, region_id: int) -> Optional[np.ndarray]:
    p = masks_root / method / f"{sample_id}_{method}_masks.pth"
    if not p.exists():
        return None
    blob = torch.load(p, map_location="cpu")
    masks = blob.get("masks", []) if isinstance(blob, dict) else []
    idx = region_id - 1
    if idx < 0 or idx >= len(masks):
        return None
    m = masks[idx]
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    arr = np.asarray(m)
    if arr.ndim != 2:
        return None
    return arr.astype(bool)


def wav_to_spec_image(wav_path: Path, sr: int, n_mels: int, n_fft: int, hop: int) -> Image.Image:
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=60.0)
    mel_norm = (mel_db + 60.0) / 60.0
    mel_color = cm.get_cmap("magma")(mel_norm)
    mel_img = (mel_color[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mel_img, mode="RGB").resize((768, 768), Image.BICUBIC)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def upscale_if_needed(img: Image.Image, min_side: int) -> Image.Image:
    w, h = img.size
    ms = min(w, h)
    if ms >= min_side or ms <= 0:
        return img
    s = float(min_side) / float(ms)
    return img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)


def draw_label(img: Image.Image, text: str) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    fs = max(28, int(min(w, h) * 0.16))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fs)
    except Exception:
        font = ImageFont.load_default()
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x, y = max(8, int(w * 0.03)), max(8, int(h * 0.03))
    d.rectangle([x - 6, y - 4, x + tw + 6, y + th + 4], fill=(0, 0, 0, 150))
    d.text((x, y), text, fill=(255, 255, 255, 255), font=font)


def main() -> None:
    args = parse_args()

    rows = read_table_rows(Path(args.table_csv))
    fake_to_real = read_pairs_map(Path(args.pairs_csv))
    masks_root = Path(args.masks_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    spec_cache: Dict[str, Image.Image] = {}

    spec_index: Dict[str, Path] = {}
    if args.real_spec_root:
        spec_root = Path(args.real_spec_root)
        spec_index = build_spec_index(spec_root, args.real_spec_suffix)
        print(f"real_spec_indexed={len(spec_index)} from {spec_root}")

    written = 0
    missing_pair = 0
    missing_real = 0
    missing_mask = 0
    missing_spec = 0

    for sample_id, method, region_id in rows:
        real_path = fake_to_real.get(sample_id)
        if real_path is None:
            missing_pair += 1
            continue

        mask = load_region_mask(masks_root, sample_id, method, region_id)
        if mask is None:
            missing_mask += 1
            continue

        if args.real_spec_root:
            spec_path = spec_index.get(real_path.stem)
            if spec_path is None:
                missing_spec += 1
                continue
            key = str(spec_path)
            if key not in spec_cache:
                spec_cache[key] = Image.open(spec_path).convert("RGB")
            img = spec_cache[key]
        else:
            if not real_path.exists():
                missing_real += 1
                continue
            key = str(real_path)
            if key not in spec_cache:
                spec_cache[key] = wav_to_spec_image(real_path, args.sr, args.n_mels, args.n_fft, args.hop)
            img = spec_cache[key]

        if img.size[::-1] != mask.shape:
            img = img.resize((mask.shape[1], mask.shape[0]), Image.BICUBIC)

        rgb = np.array(img, dtype=np.uint8)
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
            draw_label(crop, f"{args.label_prefix}{region_id}")

        method_dir = out_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        out_path = method_dir / f"{sample_id}__r{region_id}.png"
        if out_path.exists() and not args.overwrite:
            continue
        crop.save(out_path)
        written += 1

    print(f"rows_total={len(rows)}")
    print(f"written={written}")
    print(f"missing_pair={missing_pair}")
    print(f"missing_real={missing_real}")
    print(f"missing_mask={missing_mask}")
    print(f"missing_spec={missing_spec}")
    print(f"output_dir={out_root}")


if __name__ == "__main__":
    main()
