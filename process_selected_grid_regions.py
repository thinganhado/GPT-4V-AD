#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import librosa
import matplotlib.cm as cm
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d

EPS = 1e-8
SELECTED_CSV = Path("/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__En/union_all3_only.cleaned.v2.top4_distinct_vocoders.replaced_en.csv")
PAIRS_CSV = Path("/datasets/work/dss-deepfake-audio/work/data/datasets/pairs_vocv4.csv")
OUTPUT_ROOT = Path("/datasets/work/dss-deepfake-audio/work/data/datasets/gridex_demo/")
IMG_SIZE = 768
DIV_NUM = 4
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
WIN_LENGTH = 1024
CENTER = True
GAUSS_SIZE_T = 3
GAUSS_SIZE_F = 11
GAUSS_VAR_T = 3.0
GAUSS_VAR_F = 5.0
THRESH_QUANTILE = 0.95
LABEL_MODE = "region"
FLIP_STEP4_MASK_VERTICAL = True


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    last_exc = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", newline="", encoding=enc) as f:
                sample = f.read(4096)
                f.seek(0)
                first_line = sample.splitlines()[0] if sample else ""
                delimiter = "\t" if "\t" in first_line and "," not in first_line else ","
                return list(csv.DictReader(f, delimiter=delimiter))
        except UnicodeDecodeError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    return []


def load_selected_regions(csv_path: Path) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for row in read_csv_rows(csv_path):
        sample_id = str(row.get("sample_id", "")).strip()
        region_id = str(row.get("region_id", "")).strip()
        if not sample_id or not region_id:
            continue
        grouped.setdefault(sample_id, []).append(int(region_id))
    grouped = {k: v for k, v in grouped.items() if v}
    if not grouped:
        raise SystemExit(f"No sample_id/region_id rows found in {csv_path}")
    return grouped


def load_pairs_index(csv_path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(csv_path)
    index: Dict[str, Dict[str, str]] = {}
    for row in rows:
        fake_path = str(row.get("fake_path", "")).strip()
        real_path = str(row.get("real_path", "")).strip()
        if not fake_path or not real_path:
            continue
        index[Path(fake_path).stem] = row
    if not index:
        raise SystemExit(f"No fake_path/real_path rows found in {csv_path}")
    return index


def vocoder_stem(sample_id: str) -> str:
    if "_LA_" in sample_id:
        return sample_id.split("_LA_", 1)[0]
    return sample_id.split("_", 1)[0]


def audio_to_spec_png(audio_path: Path, out_path: Path, sr: int, n_mels: int, n_fft: int, hop: int, size: int) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=60.0)
    mel_norm = (mel_db + 60.0) / 60.0
    mel_color = cm.get_cmap("magma")(mel_norm)
    mel_img = (mel_color[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
    mel_img = np.flipud(mel_img)
    img = np.array(Image.fromarray(mel_img, mode="RGB").resize((size, size), Image.BICUBIC))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(out_path)
    return img


def build_grid_masks(img_size: int, div_num: int) -> List[np.ndarray]:
    if img_size % div_num != 0:
        raise ValueError(f"img_size={img_size} must be divisible by div_num={div_num}")
    div_size = img_size // div_num
    masks: List[np.ndarray] = []
    for i in range(div_num):
        for j in range(div_num):
            mask = np.zeros((img_size, img_size), dtype=bool)
            y1, y2 = i * div_size, (i + 1) * div_size
            x1, x2 = j * div_size, (j + 1) * div_size
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
    return masks


def save_grid_masks(out_path: Path, masks: Sequence[np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"masks": [m.astype(bool) for m in masks]}, out_path)


def draw_grid_lines(rgb: np.ndarray, div_num: int, color: Tuple[int, int, int] = (220, 220, 220), thickness: int = 1) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    for i in range(1, div_num):
        x = int(round(i * w / div_num))
        y = int(round(i * h / div_num))
        cv2.line(out, (x, 0), (x, h - 1), color, thickness, cv2.LINE_AA)
        cv2.line(out, (0, y), (w - 1, y), color, thickness, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (30, 30, 30), 2)
    return out


def blend_uniform(rgb: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    base = rgb.astype(np.float32)
    overlay = np.full_like(base, np.array(color, dtype=np.float32))
    out = base * (1.0 - alpha) + overlay * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def render_selected_grid_view(spec_rgb: np.ndarray, selected_region_ids: Sequence[int], div_num: int, label_mode: str = "region") -> np.ndarray:
    masks = build_grid_masks(spec_rgb.shape[0], div_num)
    selected = [rid for rid in selected_region_ids if 1 <= rid <= len(masks)]
    canvas = blend_uniform(spec_rgb, (205, 208, 214), alpha=0.72)
    for order, rid in enumerate(selected, 1):
        mask = masks[rid - 1]
        cell = blend_uniform(spec_rgb, (255, 235, 120), alpha=0.68)
        canvas[mask] = cell[mask]
        ys, xs = np.where(mask)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        text = str(rid) if label_mode == "region" else chr(ord("x") + order - 1)
        cv2.putText(canvas, text, (cx - 12, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA)
    return draw_grid_lines(canvas, div_num)


def stft_mag(y: np.ndarray, n_fft: int, hop: int, win_length: int, center: bool) -> np.ndarray:
    spec = librosa.stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop, win_length=win_length, center=center, window="hann")
    return np.abs(spec).astype(np.float32, copy=False)


def _compute_truncate(size: int, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    return max(((size - 1) / 2.0) / sigma, 0.0)


def gaussian_smooth_2d_mag(mag: np.ndarray, size_t: int, size_f: int, var_t: float, var_f: float) -> np.ndarray:
    sigma_t, sigma_f = float(np.sqrt(var_t)), float(np.sqrt(var_f))
    out = mag
    if sigma_t > 0.0:
        out = gaussian_filter1d(out, sigma=sigma_t, axis=1, mode="nearest", truncate=max(_compute_truncate(size_t, sigma_t), 1e-6))
    if sigma_f > 0.0:
        out = gaussian_filter1d(out, sigma=sigma_f, axis=0, mode="nearest", truncate=max(_compute_truncate(size_f, sigma_f), 1e-6))
    return out


def load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    y, sr = sf.read(path)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y


def compute_diff_mask(real_path: Path, fake_path: Path, sr: int, n_fft: int, hop: int, win_length: int, center: bool, gauss_size_t: int, gauss_size_f: int, gauss_var_t: float, gauss_var_f: float, thresh_quantile: float) -> Tuple[np.ndarray, np.ndarray]:
    y_real = load_audio_mono(real_path, sr)
    y_fake = load_audio_mono(fake_path, sr)
    real_mag = stft_mag(y_real, n_fft=n_fft, hop=hop, win_length=win_length, center=center)
    fake_mag = stft_mag(y_fake, n_fft=n_fft, hop=hop, win_length=win_length, center=center)
    t = min(real_mag.shape[1], fake_mag.shape[1])
    real_mag = real_mag[:, :t]
    fake_mag = fake_mag[:, :t]
    smooth_real = gaussian_smooth_2d_mag(real_mag, gauss_size_t, gauss_size_f, gauss_var_t, gauss_var_f)
    smooth_fake = gaussian_smooth_2d_mag(fake_mag, gauss_size_t, gauss_size_f, gauss_var_t, gauss_var_f)
    norm_diff = np.abs(smooth_fake - smooth_real) / (smooth_real + EPS)
    finite = norm_diff[np.isfinite(norm_diff)]
    tau = float(np.quantile(finite, thresh_quantile)) if finite.size else 0.0
    mask = norm_diff > tau
    return mask.astype(bool), smooth_fake


def to_fake_gray_image(fake_mag: np.ndarray, size: int) -> np.ndarray:
    db = librosa.amplitude_to_db(np.maximum(fake_mag, EPS), ref=np.max)
    db = np.clip(db, -60.0, 0.0)
    lo = np.percentile(db, 5.0)
    hi = np.percentile(db, 99.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -60.0, 0.0
    z = np.clip((db - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    disp = 0.62 + (0.96 - 0.62) * z
    disp = np.flipud((disp * 255.0).clip(0, 255).astype(np.uint8))
    img = np.array(Image.fromarray(disp, mode="L").resize((size, size), Image.BICUBIC))
    return np.stack([img, img, img], axis=-1)


def resize_bool_mask(mask: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.array(img.resize((size, size), Image.NEAREST)) > 0


def overlay_red_mask(base_rgb: np.ndarray, red_mask: np.ndarray, alpha: float) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    out = base.copy()
    red = np.array([220.0, 0.0, 0.0], dtype=np.float32)
    out[red_mask] = base[red_mask] * (1.0 - alpha) + red * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def render_selected_mask_view(fake_mag: np.ndarray, diff_mask: np.ndarray, selected_region_ids: Sequence[int], div_num: int, size: int) -> np.ndarray:
    base = to_fake_gray_image(fake_mag, size)
    if FLIP_STEP4_MASK_VERTICAL:
        diff_mask = np.flipud(diff_mask)
    diff_mask_resized = resize_bool_mask(diff_mask, size)
    masks = build_grid_masks(size, div_num)
    selected_union = np.zeros((size, size), dtype=bool)
    for rid in selected_region_ids:
        if 1 <= rid <= len(masks):
            selected_union |= masks[rid - 1]
    background = blend_uniform(base, (165, 165, 165), alpha=0.45)
    canvas = background.copy()
    canvas[selected_union] = base[selected_union]
    selected_red = diff_mask_resized & selected_union
    canvas = overlay_red_mask(canvas, selected_red, alpha=0.92)
    canvas = draw_grid_lines(canvas, div_num)
    for rid in selected_region_ids:
        if not (1 <= rid <= len(masks)):
            continue
        mask = masks[rid - 1]
        ys, xs = np.where(mask)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (230, 0, 0), 3)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        cv2.putText(canvas, str(rid), (cx - 12, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 0, 0), 2, cv2.LINE_AA)
    return canvas


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    selected_csv = SELECTED_CSV.expanduser().resolve()
    pairs_csv = PAIRS_CSV.expanduser().resolve()
    out_root = OUTPUT_ROOT.expanduser().resolve()
    selected = load_selected_regions(selected_csv)
    pairs_index = load_pairs_index(pairs_csv)

    step1_dir = out_root / "step1_spectrogram"
    step2_dir = out_root / "step2_grid"
    step3_dir = out_root / "step3_selected_grid"
    step4_dir = out_root / "step4_selected_mask"
    meta_dir = out_root / "meta"
    for d in (step1_dir, step2_dir, step3_dir, step4_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    missing = [sid for sid in sorted(selected) if sid not in pairs_index]
    if missing:
        raise SystemExit(f"Missing sample_ids in pairs CSV: {missing}")

    for sample_id in sorted(selected):
        row = pairs_index[sample_id]
        fake_path = Path(str(row["fake_path"]).strip()).expanduser()
        real_path = Path(str(row["real_path"]).strip()).expanduser()
        vocoder = vocoder_stem(sample_id)

        spec_path = step1_dir / f"{vocoder}.png"
        spec_rgb = audio_to_spec_png(fake_path, spec_path, SR, N_MELS, N_FFT, HOP, IMG_SIZE)

        grid_masks = build_grid_masks(IMG_SIZE, DIV_NUM)
        save_grid_masks(meta_dir / f"{sample_id}_grid_masks.pth", grid_masks)
        save_rgb(step2_dir / f"{vocoder}.png", draw_grid_lines(spec_rgb, DIV_NUM))

        selected_grid = render_selected_grid_view(spec_rgb, selected[sample_id], DIV_NUM, label_mode=LABEL_MODE)
        save_rgb(step3_dir / f"{vocoder}.png", selected_grid)

        diff_mask, fake_mag = compute_diff_mask(
            real_path=real_path,
            fake_path=fake_path,
            sr=SR,
            n_fft=N_FFT,
            hop=HOP,
            win_length=WIN_LENGTH,
            center=CENTER,
            gauss_size_t=GAUSS_SIZE_T,
            gauss_size_f=GAUSS_SIZE_F,
            gauss_var_t=GAUSS_VAR_T,
            gauss_var_f=GAUSS_VAR_F,
            thresh_quantile=THRESH_QUANTILE,
        )
        diff_mask_to_save = np.flipud(diff_mask) if FLIP_STEP4_MASK_VERTICAL else diff_mask
        Image.fromarray((resize_bool_mask(diff_mask_to_save, IMG_SIZE).astype(np.uint8) * 255), mode="L").save(meta_dir / f"{sample_id}_diff_mask.png")
        selected_mask = render_selected_mask_view(fake_mag, diff_mask, selected[sample_id], DIV_NUM, IMG_SIZE)
        save_rgb(step4_dir / f"{vocoder}.png", selected_mask)

        print(f"[OK] {sample_id} -> {vocoder}")


if __name__ == "__main__":
    main()
