#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import csv
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8
EXPECTED_LEN = 64600


def find_pairs_in_dirs(real_dir: Path, fake_dir: Path, exts=(".wav", ".flac")) -> List[Tuple[Path, Path]]:
    real_index = {p.stem: p for p in real_dir.rglob("*") if p.suffix.lower() in exts}
    pairs = []
    for q in fake_dir.rglob("*"):
        if q.suffix.lower() in exts and q.stem in real_index:
            pairs.append((real_index[q.stem], q))
    return sorted(pairs)


def read_pairs_from_csv(csv_path: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel_tab if "\t" in sample else csv.excel
        reader = csv.reader(f, dialect)
        for row in reader:
            if not row or row[0].startswith("#") or row[0].lower() == "real_path":
                continue
            if len(row) >= 2:
                rp_s = row[0].strip().strip('"').strip("'")
                fp_s = row[1].strip().strip('"').strip("'")
                rp, fp = Path(rp_s).expanduser(), Path(fp_s).expanduser()
                if rp.exists() and fp.exists():
                    pairs.append((rp, fp))
    return pairs


def pad_or_crop_1d_np(y: np.ndarray, length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    n = y.shape[0]
    if n < length:
        return np.pad(y, (0, length - n), mode="constant")
    if n > length:
        return y[:length]
    return y


_HANN = None


def stft_mag_torch(
    y: np.ndarray,
    n_fft=1024,
    hop=256,
    win_length=1024,
    center=True,
    device: str = DEVICE,
):
    global _HANN
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    if _HANN is None or _HANN.device != yt.device or _HANN.numel() != win_length:
        _HANN = torch.hann_window(win_length, periodic=True, device=yt.device)
    with torch.no_grad():
        S = torch.stft(
            yt,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_length,
            window=_HANN,
            center=center,
            return_complex=False,
        )
        real = S[..., 0]
        imag = S[..., 1]
        mag = torch.sqrt(real * real + imag * imag).to("cpu").numpy().astype(np.float32, copy=False)
    return mag


def _compute_truncate(size: int, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    radius = max((size - 1) / 2.0, 0.0)
    return max(radius / sigma, 0.0)


def gaussian_smooth_2d_mag(M, size_t=3, size_f=11, var_t=3.0, var_f=5.0):
    if size_t % 2 == 0 or size_f % 2 == 0:
        raise ValueError("Gaussian kernel sizes must be odd to preserve alignment.")
    sigma_t, sigma_f = math.sqrt(var_t), math.sqrt(var_f)
    Mt = M
    if sigma_t > 0.0:
        trunc_t = _compute_truncate(size_t, sigma_t)
        Mt = gaussian_filter1d(Mt, sigma=sigma_t, axis=1, mode="nearest", truncate=max(trunc_t, 1e-6))
    if sigma_f > 0.0:
        trunc_f = _compute_truncate(size_f, sigma_f)
        Mt = gaussian_filter1d(Mt, sigma=sigma_f, axis=0, mode="nearest", truncate=max(trunc_f, 1e-6))
    return Mt


def align_mel_with_dtw(Mb_mel, Ms_mel):
    X = np.log(np.maximum(Mb_mel, EPS))
    Y = np.log(np.maximum(Ms_mel, EPS))
    _D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    wp = wp[::-1]
    return Mb_mel[:, wp[:, 0]], Ms_mel[:, wp[:, 1]]


def compute_mask_from_pair_mel(
    y_bona,
    y_spoof,
    sr=16000,
    n_fft=1024,
    hop=256,
    win_length=1024,
    center=True,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    gauss_size_t=3,
    gauss_size_f=11,
    gauss_var_t=3.0,
    gauss_var_f=5.0,
    use_dtw=False,
    thresh_quantile=0.95,
):
    y_bona = pad_or_crop_1d_np(y_bona, EXPECTED_LEN)
    y_spoof = pad_or_crop_1d_np(y_spoof, EXPECTED_LEN)

    Mb_mag = stft_mag_torch(y_bona, n_fft, hop, win_length, center, device=DEVICE)
    Ms_mag = stft_mag_torch(y_spoof, n_fft, hop, win_length, center, device=DEVICE)

    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    Mb_mel = mel @ (Mb_mag**2)
    Ms_mel = mel @ (Ms_mag**2)

    if use_dtw:
        Mb_mel, Ms_mel = align_mel_with_dtw(Mb_mel, Ms_mel)
    else:
        T = min(Mb_mel.shape[1], Ms_mel.shape[1])
        Mb_mel, Ms_mel = Mb_mel[:, :T], Ms_mel[:, :T]

    G_Mb = gaussian_smooth_2d_mag(Mb_mel, gauss_size_t, gauss_size_f, gauss_var_t, gauss_var_f)
    G_Ms = gaussian_smooth_2d_mag(Ms_mel, gauss_size_t, gauss_size_f, gauss_var_t, gauss_var_f)

    diff = np.abs(G_Ms - G_Mb)
    norm_diff = diff / (np.abs(G_Mb) + EPS)

    finite = norm_diff[np.isfinite(norm_diff)]
    tau = float(np.quantile(finite, thresh_quantile)) if finite.size else 0.0
    mask = (norm_diff > tau).astype(np.uint8)

    return mask, norm_diff.astype(np.float32, copy=False), tau


def resize_mask(mask, size):
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return img.resize(size, Image.NEAREST)


def overlay_mask_on_image(base_img: Image.Image, mask_img: Image.Image, alpha: float, color):
    base = base_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (color[0], color[1], color[2], 0))
    mask_alpha = mask_img.point(lambda p: int(p * alpha))
    overlay.putalpha(mask_alpha)
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")

def _load_region_masks(pth_path: Path):
    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    masks = data.get("masks", [])
    return [np.asarray(m, dtype=bool) for m in masks]


def main():
    p = argparse.ArgumentParser(description="Overlay artifact masks on spectrogram images.")
    m = p.add_mutually_exclusive_group(required=False)
    m.add_argument("--real_dir", "--real-dir", dest="real_dir", type=str, help="Directory of real audios.")
    m.add_argument("--input_pairs", "--pairs-csv", dest="input_pairs", type=str, default="/scratch3/che489/Ha/AudioDeepfake-XAI/scripts/pairs_vocv4_full.csv")
    p.add_argument("--fake_dir", "--fake-dir", dest="fake_dir", type=str, help="Directory of fake audios.")
    p.add_argument("--spec_dir", "--spec-dir", dest="spec_dir", type=str, default="/scratch3/che489/Ha/interspeech/localization/specs")
    p.add_argument("--spec_suffix", "--spec-suffix", dest="spec_suffix", type=str, default=".png")
    p.add_argument("--output_dir", "--out-dir", dest="output_dir", type=str, required=True)
    p.add_argument("--region_outputs", "--region-outputs", dest="region_outputs", type=str, default="/scratch3/che489/Ha/interspeech/localization/Ms_region_outputs")
    p.add_argument("--region_methods", "--region-methods", dest="region_methods", nargs="+", default=["grid", "superpixel", "sam"])
    p.add_argument("--overlay_from_regions", "--overlay-from-regions", dest="overlay_from_regions", action="store_true")
    p.add_argument("--overlay_region_method", "--overlay-region-method", dest="overlay_region_method", type=str, default="grid")
    p.add_argument("--overlay_region_suffix", "--overlay-region-suffix", dest="overlay_region_suffix", type=str, default="_img_edge_number.png")

    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--center", action="store_true")
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--fmin", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=None)

    p.add_argument("--gauss_size_t", type=int, default=3)
    p.add_argument("--gauss_size_f", type=int, default=11)
    p.add_argument("--gauss_var_t", type=float, default=3.0)
    p.add_argument("--gauss_var_f", type=float, default=5.0)
    p.add_argument("--use_dtw", action="store_true")
    p.add_argument("--thresh_quantile", type=float, default=0.95)

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--img_size", type=int, default=768)
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--overlay_color", "--overlay-color", dest="overlay_color", type=str, default="0,0,255")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--save_mask", action="store_true", default=False)

    args = p.parse_args()

    if not args.center:
        args.center = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = Path(args.spec_dir)
    region_root = Path(args.region_outputs)

    if args.input_pairs:
        pairs = read_pairs_from_csv(Path(args.input_pairs))
    else:
        if not args.real_dir or not args.fake_dir:
            raise ValueError("Both --real_dir and --fake_dir are required when CSV pairs are not provided.")
        pairs = find_pairs_in_dirs(Path(args.real_dir), Path(args.fake_dir))

    if args.limit is not None:
        pairs = pairs[: args.limit]

    filtered = []
    for r, f in pairs:
        stem = Path(f).stem
        missing = []
        for method in args.region_methods:
            pth_path = region_root / method / f"{stem}_{method}_masks.pth"
            if not pth_path.exists():
                missing.append(method)
        if missing:
            continue
        filtered.append((r, f))
    pairs = filtered

    print(f"[INFO] Found {len(pairs)} pairs -> {out_dir}")

    rows = []
    for i, (r, f) in enumerate(pairs, 1):
        try:
            yb, srb = sf.read(r)
            ys, srs = sf.read(f)

            if isinstance(yb, np.ndarray) and yb.ndim > 1:
                yb = yb.mean(axis=1)
            if isinstance(ys, np.ndarray) and ys.ndim > 1:
                ys = ys.mean(axis=1)

            if srb != args.sr:
                yb = librosa.resample(yb, orig_sr=srb, target_sr=args.sr)
            if srs != args.sr:
                ys = librosa.resample(ys, orig_sr=srs, target_sr=args.sr)

            mask, norm_diff, tau = compute_mask_from_pair_mel(
                yb,
                ys,
                sr=args.sr,
                n_fft=args.n_fft,
                hop=args.hop,
                win_length=args.win_length,
                center=args.center,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=args.fmax,
                gauss_size_t=args.gauss_size_t,
                gauss_size_f=args.gauss_size_f,
                gauss_var_t=args.gauss_var_t,
                gauss_var_f=args.gauss_var_f,
                use_dtw=args.use_dtw,
                thresh_quantile=args.thresh_quantile,
            )

            stem = Path(f).stem
            spec_path = spec_dir / f"{stem}{args.spec_suffix}"
            if not spec_path.exists():
                print(f"[WARN] Missing spec for {stem}: {spec_path}")
                continue

            if args.overlay_from_regions:
                region_img = (
                    region_root
                    / args.overlay_region_method
                    / f"{stem}_{args.overlay_region_method}{args.overlay_region_suffix}"
                )
                if not region_img.exists():
                    print(f"[WARN] Missing region image for overlay: {region_img}")
                    continue
                base_img = Image.open(region_img).convert("RGB").resize((args.img_size, args.img_size), Image.BICUBIC)
            else:
                base_img = Image.open(spec_path).convert("RGB").resize((args.img_size, args.img_size), Image.BICUBIC)
            mask_img = resize_mask(mask, base_img.size)
            color_parts = [int(c) for c in args.overlay_color.split(",")]
            out_path = out_dir / f"{stem}_diff_overlay.png"
            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] {stem} overlay exists -> {out_path}")
                continue
            out_img = overlay_mask_on_image(base_img, mask_img, args.alpha, color_parts)
            out_img.save(out_path)

            if args.save_mask:
                mask_img.save(out_dir / f"{stem}_diff_mask.png")

            diff_mask = np.array(mask_img) > 0
            for method in args.region_methods:
                pth_path = region_root / method / f"{stem}_{method}_masks.pth"
                if not pth_path.exists():
                    print(f"[WARN] Missing region masks for {stem} ({method}): {pth_path}")
                    continue
                region_masks = _load_region_masks(pth_path)
                for ridx, region_mask in enumerate(region_masks, 1):
                    if region_mask.shape != diff_mask.shape:
                        region_img = Image.fromarray(region_mask.astype(np.uint8) * 255, mode="L")
                        region_mask = np.array(region_img.resize(diff_mask.shape[::-1], Image.NEAREST)) > 0
                    region_pixels = int(region_mask.sum())
                    if region_pixels == 0:
                        continue
                    overlap_pixels = int(np.logical_and(region_mask, diff_mask).sum())
                    coverage = float(overlap_pixels) / float(region_pixels)
                    present = 1 if overlap_pixels > 0 else 0
                    rows.append(
                        {
                            "image": stem,
                            "method": method,
                            "region_id": ridx,
                            "region_pixels": region_pixels,
                            "overlap_pixels": overlap_pixels,
                            "coverage": coverage,
                            "present": present,
                        }
                    )

            print(f"[OK] {i:04d}/{len(pairs):04d} {stem} tau={tau:.6f}")

        except Exception as e:
            print(f"[ERR] {i:04d}/{len(pairs):04d} {Path(r).name}->{Path(f).name}: {e}")

    if rows:
        csv_path = out_dir / "region_diff_stats.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "method",
                    "region_id",
                    "region_pixels",
                    "overlap_pixels",
                    "coverage",
                    "present",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Wrote region overlap stats -> {csv_path}")

    print("[DONE] All pairs processed.")


if __name__ == "__main__":
    main()
