#!/usr/bin/env python3
"""
Build a tabular region metadata file with columns:
sample_id, method, region_id, T, F, P, diff

Inputs:
- region masks from region_division.py outputs (*_masks.pth)
- MFA alignment JSON files (tiers -> phones -> entries)
- overlay masks (e.g., *_gt_binary.npy from make_masks_overlay.py)
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import torch
except ModuleNotFoundError:
    torch = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create per-region T/F/P/diff table.")
    p.add_argument("--masks-root", type=str, required=True, help="Root containing <method>/*_masks.pth.")
    p.add_argument("--mfa-json-root", type=str, required=True, help="Folder with MFA JSON files.")
    p.add_argument("--overlay-root", type=str, required=True, help="Folder with overlay .npy masks.")
    p.add_argument("--output-csv", type=str, required=True, help="Output CSV path.")
    p.add_argument("--output-parquet", type=str, default=None, help="Optional output parquet path.")
    p.add_argument("--method-glob", type=str, default="*", help="Method folder glob under masks-root.")
    p.add_argument(
        "--overlay-mode",
        choices=["auto", "binary", "continuous"],
        default="auto",
        help="How to pick overlay files.",
    )
    p.add_argument("--overlay-binary-suffix", type=str, default="_gt_binary.npy")
    p.add_argument("--overlay-continuous-suffix", type=str, default="_gt_continuous.npy")
    p.add_argument(
        "--continuous-threshold",
        type=float,
        default=0.5,
        help="Threshold if using continuous overlay map.",
    )
    p.add_argument(
        "--min-speech-overlap-sec",
        type=float,
        default=0.02,
        help="If max phone overlap below this, set T=non_speech and P=none.",
    )
    p.add_argument(
        "--f-axis-top-is-high",
        action="store_true",
        default=True,
        help="Assume top rows are high frequency (default true for common spectrogram images).",
    )
    return p.parse_args()


def infer_sample_id(mask_path: str, method: str) -> str:
    stem = Path(mask_path).name.replace("_masks.pth", "")
    suffix = f"_{method}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def infer_canonical_stem(sample_id: str) -> str:
    if "_LA_" in sample_id:
        return "LA_" + sample_id.split("_LA_", 1)[1]
    return sample_id


def to_bool_array(mask) -> np.ndarray:
    if torch is None:
        raise SystemExit("Missing dependency: torch")
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={arr.shape}")
    return arr.astype(bool)


def load_masks(mask_path: str) -> List[np.ndarray]:
    if torch is None:
        raise SystemExit("Missing dependency: torch")
    blob = torch.load(mask_path, map_location="cpu")
    if not isinstance(blob, dict) or "masks" not in blob:
        raise ValueError(f"Unexpected mask file format: {mask_path}")
    return [to_bool_array(m) for m in blob["masks"]]


def load_mfa_json(mfa_json_root: str, sample_id: str) -> Optional[dict]:
    p = Path(mfa_json_root) / f"{sample_id}.json"
    if not p.exists():
        return None
    import json

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_phone_entries(mfa_obj: dict) -> List[Tuple[float, float, str]]:
    tiers = mfa_obj.get("tiers", {})
    phones = tiers.get("phones", {})
    entries = phones.get("entries", [])
    out: List[Tuple[float, float, str]] = []
    for e in entries:
        if isinstance(e, (list, tuple)) and len(e) >= 3:
            start, end, label = float(e[0]), float(e[1]), str(e[2]).strip()
            if label:
                out.append((start, end, label))
    return out


def get_audio_end_sec(mfa_obj: dict, phone_entries: Sequence[Tuple[float, float, str]]) -> float:
    end = mfa_obj.get("end", None)
    if end is not None:
        return float(end)
    if phone_entries:
        return float(max(e for _, e, _ in phone_entries))
    return 0.0


def find_overlay_path(args: argparse.Namespace, sample_id: str) -> Optional[Path]:
    root = Path(args.overlay_root)
    candidates = [sample_id, infer_canonical_stem(sample_id)]
    for key in candidates:
        b = root / f"{key}{args.overlay_binary_suffix}"
        c = root / f"{key}{args.overlay_continuous_suffix}"
        if args.overlay_mode == "binary":
            if b.exists():
                return b
        elif args.overlay_mode == "continuous":
            if c.exists():
                return c
        else:
            if b.exists():
                return b
            if c.exists():
                return c
    return None


def load_overlay_bool(path: Path, args: argparse.Namespace) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Overlay mask must be 2D: {path}")
    if path.name.endswith(args.overlay_binary_suffix):
        return arr > 0
    return arr >= args.continuous_threshold


def resize_nn(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = arr.shape
    if (h, w) == (out_h, out_w):
        return arr
    y_idx = np.clip((np.arange(out_h) * h / out_h).astype(int), 0, h - 1)
    x_idx = np.clip((np.arange(out_w) * w / out_w).astype(int), 0, w - 1)
    return arr[np.ix_(y_idx, x_idx)]


def region_time_window(mask: np.ndarray, audio_end_sec: float) -> Tuple[float, float]:
    h, w = mask.shape
    if w <= 1:
        return 0.0, audio_end_sec
    x_idx = np.where(mask.any(axis=0))[0]
    if len(x_idx) == 0:
        return 0.0, 0.0
    x_min, x_max = int(x_idx.min()), int(x_idx.max())
    t0 = (x_min / (w - 1)) * audio_end_sec
    t1 = (x_max / (w - 1)) * audio_end_sec
    return float(t0), float(t1)


def dominant_phone(
    t0: float, t1: float, phone_entries: Sequence[Tuple[float, float, str]], min_overlap: float
) -> Tuple[str, str]:
    if t1 <= t0 or len(phone_entries) == 0:
        return "non_speech", "none"
    best_label = "none"
    best_overlap = 0.0
    for p0, p1, label in phone_entries:
        overlap = max(0.0, min(t1, p1) - max(t0, p0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    if best_overlap < min_overlap:
        return "non_speech", "none"
    return "speech", best_label


def f_band(mask: np.ndarray, top_is_high: bool = True) -> str:
    h, _ = mask.shape
    y_idx = np.where(mask.any(axis=1))[0]
    if len(y_idx) == 0:
        return "mid"
    y_c = float(np.mean(y_idx))
    third = h / 3.0
    if top_is_high:
        if y_c < third:
            return "high"
        if y_c < 2 * third:
            return "mid"
        return "low"
    if y_c < third:
        return "low"
    if y_c < 2 * third:
        return "mid"
    return "high"


def collect_mask_files(masks_root: str, method_glob: str) -> List[Tuple[str, str]]:
    pattern = os.path.join(masks_root, method_glob, "*_masks.pth")
    files = sorted(glob.glob(pattern))
    out = []
    for p in files:
        method = Path(p).parent.name
        out.append((method, p))
    return out


def main() -> None:
    args = parse_args()
    if pd is None:
        raise SystemExit("Missing dependency: pandas")
    if torch is None:
        raise SystemExit("Missing dependency: torch")
    pairs = collect_mask_files(args.masks_root, args.method_glob)
    if len(pairs) == 0:
        raise SystemExit("No *_masks.pth files found.")

    rows = []
    missing_mfa = 0
    missing_overlay = 0

    for method, mask_path in pairs:
        sample_id = infer_sample_id(mask_path, method)
        mfa_obj = load_mfa_json(args.mfa_json_root, sample_id)
        if mfa_obj is None:
            missing_mfa += 1
            continue

        phone_entries = get_phone_entries(mfa_obj)
        audio_end_sec = get_audio_end_sec(mfa_obj, phone_entries)
        masks = load_masks(mask_path)
        if len(masks) == 0:
            continue

        overlay_path = find_overlay_path(args, sample_id)
        if overlay_path is None:
            missing_overlay += 1
            continue
        overlay_bool = load_overlay_bool(overlay_path, args)
        h, w = masks[0].shape
        overlay_bool = resize_nn(overlay_bool.astype(bool), h, w)

        for idx, mask in enumerate(masks, start=1):
            t0, t1 = region_time_window(mask, audio_end_sec)
            t_label, p_label = dominant_phone(
                t0, t1, phone_entries=phone_entries, min_overlap=args.min_speech_overlap_sec
            )
            freq_band = f_band(mask, top_is_high=args.f_axis_top_is_high)
            diff = int(np.logical_and(mask, overlay_bool).sum())

            rows.append(
                {
                    "sample_id": sample_id,
                    "method": method,
                    "region_id": idx,
                    "T": t_label,
                    "F": freq_band,
                    "P": p_label,
                    "diff": diff,
                }
            )

    df = pd.DataFrame(rows, columns=["sample_id", "method", "region_id", "T", "F", "P", "diff"])
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if args.output_parquet:
        out_parquet = Path(args.output_parquet)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)

    print(f"rows_written={len(df)}")
    print(f"unique_samples={df['sample_id'].nunique() if len(df) else 0}")
    print(f"missing_mfa_json={missing_mfa}")
    print(f"missing_overlay={missing_overlay}")
    print(f"output_csv={out_csv}")
    if args.output_parquet:
        print(f"output_parquet={args.output_parquet}")


if __name__ == "__main__":
    main()
