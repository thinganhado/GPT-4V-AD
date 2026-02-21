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
import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8


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
    # Match reference logic:
    # trunc = ((ksize - 1) / 2) / sigma  so effective window equals requested odd size.
    return max(((size - 1) / 2.0) / sigma, 0.0)


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


def compute_mask_from_pair_magnitude(
    y_bona,
    y_spoof,
    sr=16000,
    n_fft=1024,
    hop=256,
    win_length=1024,
    center=True,
    gauss_size_t=3,
    gauss_size_f=11,
    gauss_var_t=3.0,
    gauss_var_f=5.0,
    use_dtw=False,
    thresh_quantile=0.95,
):
    # Keep natural lengths; align later by min-T or DTW.
    Mb_mag = stft_mag_torch(y_bona, n_fft, hop, win_length, center, device=DEVICE)
    Ms_mag = stft_mag_torch(y_spoof, n_fft, hop, win_length, center, device=DEVICE)

    if use_dtw:
        Mb_mag, Ms_mag = align_mel_with_dtw(Mb_mag, Ms_mag)
    else:
        T = min(Mb_mag.shape[1], Ms_mag.shape[1])
        Mb_mag, Ms_mag = Mb_mag[:, :T], Ms_mag[:, :T]

    G_Mb = gaussian_smooth_2d_mag(Mb_mag, gauss_size_t, gauss_size_f, gauss_var_t, gauss_var_f)
    G_Ms = gaussian_smooth_2d_mag(Ms_mag, gauss_size_t, gauss_size_f, gauss_var_t, gauss_var_f)

    diff = np.abs(G_Ms - G_Mb)
    # Match reference Eq. on linear magnitude.
    norm_diff = diff / (G_Mb + EPS)

    finite = norm_diff[np.isfinite(norm_diff)]
    tau = float(np.quantile(finite, thresh_quantile)) if finite.size else 0.0
    mask = (norm_diff > tau).astype(np.uint8)

    return mask, norm_diff.astype(np.float32, copy=False), tau, Mb_mag, Ms_mag, G_Mb, G_Ms


def resize_mask(mask, size):
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return img.resize(size, Image.NEAREST)


def overlay_mask_on_image(base_img: Image.Image, mask_img: Image.Image, alpha: float, color, grayscale_base: bool = False):
    if grayscale_base:
        base = base_img.convert("L").convert("RGB").convert("RGBA")
    else:
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


RED_MASK_CMAP = ListedColormap(
    [
        [0.0, 0.0, 0.0, 0.0],  # transparent
        [0.75, 0.0, 0.0, 1.0],  # red
    ]
)


def _extent(frames: int, sr: int, hop: int):
    t_max = frames * hop / float(sr)
    f_max = sr / 2.0
    return [0.0, t_max, 0.0, f_max]


def _to_db_clipped(mag: np.ndarray) -> np.ndarray:
    db = librosa.amplitude_to_db(np.maximum(mag, EPS), ref=np.max)
    return np.clip(db, -60.0, 0.0)


def paper_gray_from_db(
    db_img: np.ndarray,
    lo_pct: float = 5.0,
    hi_pct: float = 99.0,
    out_lo: float = 0.62,
    out_hi: float = 0.96,
    gamma: float = 1.0,
) -> np.ndarray:
    lo = np.percentile(db_img, lo_pct)
    hi = np.percentile(db_img, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -60.0, 0.0
    z = (db_img - lo) / (hi - lo + 1e-12)
    z = np.clip(z, 0.0, 1.0) ** gamma
    return out_lo + (out_hi - out_lo) * z


def save_annotation_overlay(bg_db: np.ndarray, mask: np.ndarray, out_path: Path, sr: int, hop: int, title: str):
    ext = _extent(bg_db.shape[1], sr, hop)
    bg_disp = paper_gray_from_db(bg_db)

    plt.figure(figsize=(5.2, 4.2))
    plt.imshow(
        bg_disp,
        extent=ext,
        aspect="auto",
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    plt.imshow(
        np.ma.masked_where(mask == 0, mask.astype(float)),
        extent=ext,
        aspect="auto",
        origin="lower",
        cmap=RED_MASK_CMAP,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def save_quicklook_paper_style(
    out_path: Path,
    G_Mb_db: np.ndarray,
    G_Ms_db: np.ndarray,
    mask: np.ndarray,
    sr: int,
    hop: int,
):
    ext = _extent(G_Mb_db.shape[1], sr, hop)
    disp_b = paper_gray_from_db(G_Mb_db)
    disp_s = paper_gray_from_db(G_Ms_db)

    plt.figure(figsize=(12, 3.8))
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(
        disp_b,
        extent=ext,
        aspect="auto",
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax1.set_title("Smoothed Real")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Frequency [Hz]")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(
        disp_s,
        extent=ext,
        aspect="auto",
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax2.set_title("Smoothed Fake")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(
        disp_b,
        extent=ext,
        aspect="auto",
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax3.imshow(
        np.ma.masked_where(mask == 0, mask.astype(float)),
        extent=ext,
        aspect="auto",
        origin="lower",
        cmap=RED_MASK_CMAP,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax3.set_title("Smoothed Annotation, 95 percent")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def save_single_spec_paper_style(
    out_path: Path,
    spec_db: np.ndarray,
    sr: int,
    hop: int,
    title: str,
    mask: np.ndarray = None,
    hide_axes: bool = True,
):
    ext = _extent(spec_db.shape[1], sr, hop)
    disp = paper_gray_from_db(spec_db)

    plt.figure(figsize=(6.0, 6.0))
    plt.imshow(
        disp,
        extent=ext,
        aspect="auto",
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    if mask is not None:
        plt.imshow(
            np.ma.masked_where(mask == 0, mask.astype(float)),
            extent=ext,
            aspect="auto",
            origin="lower",
            cmap=RED_MASK_CMAP,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
    if hide_axes:
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0)
    else:
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=130)
    plt.close()


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
    p.add_argument("--overlay_all_region_methods", "--overlay-all-region-methods", dest="overlay_all_region_methods", action="store_true", help="Generate one overlay per method in --region_methods.")
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
    p.add_argument(
        "--sample_ids",
        "--sample-ids",
        dest="sample_ids",
        type=str,
        default="",
        help="Optional semicolon-separated sample_id list (stems). Only these will be processed.",
    )
    p.add_argument("--img_size", type=int, default=768)
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--overlay_color", "--overlay-color", dest="overlay_color", type=str, default="0,0,255")
    p.add_argument(
        "--grayscale_base",
        "--grayscale-base",
        dest="grayscale_base",
        action="store_true",
        default=False,
        help="Convert base spectrogram image to grayscale before applying color overlay.",
    )
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--save_mask", action="store_true", default=False)
    p.add_argument(
        "--save_paper_style",
        "--save-paper-style",
        dest="save_paper_style",
        action="store_true",
        default=False,
        help="Save paper-style spectrogram/mask visualizations from spectrogram arrays.",
    )
    p.add_argument(
        "--save_raw",
        "--save-raw",
        dest="save_raw",
        action="store_true",
        default=False,
        help="Also save raw (unsmoothed) paper-style annotation view.",
    )
    p.add_argument(
        "--save_fake_spec",
        "--save-fake-spec",
        dest="save_fake_spec",
        action="store_true",
        default=False,
        help="Save standalone fake spectrogram (and fake+mask) from STFT arrays.",
    )
    p.add_argument(
        "--flip_mask_vertical",
        "--flip-mask-vertical",
        dest="flip_mask_vertical",
        action="store_true",
        default=False,
        help="Flip computed diff mask vertically before overlay/statistics.",
    )
    p.add_argument(
        "--flip_region_masks_vertical",
        "--flip-region-masks-vertical",
        dest="flip_region_masks_vertical",
        action="store_true",
        default=False,
        help="Flip region masks from *_masks.pth vertically before overlap calculation.",
    )
    p.add_argument(
        "--csv-only",
        action="store_true",
        default=False,
        help="Skip writing overlay/mask images and only compute CSV outputs.",
    )
    p.add_argument("--topk-explain-k", type=int, default=3, help="Top-k regions for explained-difference ratio.")
    p.add_argument(
        "--topk-explain-min-ratio",
        type=float,
        default=0.5,
        help="Pass threshold for top-k explained-difference ratio.",
    )

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

    if args.sample_ids.strip():
        keep = {s.strip() for s in args.sample_ids.split(";") if s.strip()}
        pairs = [(r, f) for (r, f) in pairs if Path(f).stem in keep]

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

            mask, norm_diff, tau, Mb_mag, Ms_mag, G_Mb, G_Ms = compute_mask_from_pair_magnitude(
                yb,
                ys,
                sr=args.sr,
                n_fft=args.n_fft,
                hop=args.hop,
                win_length=args.win_length,
                center=args.center,
                gauss_size_t=args.gauss_size_t,
                gauss_size_f=args.gauss_size_f,
                gauss_var_t=args.gauss_var_t,
                gauss_var_f=args.gauss_var_f,
                use_dtw=args.use_dtw,
                thresh_quantile=args.thresh_quantile,
            )

            stem = Path(f).stem
            if args.flip_mask_vertical:
                mask = np.flipud(mask)
            diff_mask = np.array(resize_mask(mask, (args.img_size, args.img_size))) > 0

            if args.save_paper_style:
                Mb_db = _to_db_clipped(Mb_mag)
                Ms_db = _to_db_clipped(Ms_mag)
                G_Mb_db = _to_db_clipped(G_Mb)
                G_Ms_db = _to_db_clipped(G_Ms)
                save_annotation_overlay(
                    G_Mb_db,
                    mask,
                    out_dir / f"smoothed_mask95__{stem}.png",
                    args.sr,
                    args.hop,
                    "Smoothed Annotation, 95 percent",
                )
                save_quicklook_paper_style(
                    out_dir / f"quicklook__{stem}.png",
                    G_Mb_db,
                    G_Ms_db,
                    mask,
                    args.sr,
                    args.hop,
                )
                if args.save_raw:
                    raw_mask = (np.abs(Ms_mag - Mb_mag) / (Mb_mag + EPS)) > tau
                    save_annotation_overlay(
                        Mb_db,
                        raw_mask.astype(np.uint8),
                        out_dir / f"raw_mask95__{stem}.png",
                        args.sr,
                        args.hop,
                        "Annotation, 95 percent",
                    )
            if args.save_fake_spec:
                Ms_db = _to_db_clipped(Ms_mag)
                G_Ms_db = _to_db_clipped(G_Ms)
                save_single_spec_paper_style(
                    out_dir / f"fake_spec__{stem}.png",
                    Ms_db,
                    args.sr,
                    args.hop,
                    "Fake Spectrogram",
                )
                save_single_spec_paper_style(
                    out_dir / f"fake_spec_with_mask__{stem}.png",
                    G_Ms_db,
                    args.sr,
                    args.hop,
                    "Smoothed Fake + Diff Mask",
                    mask=mask,
                )

            if not args.csv_only:
                spec_path = spec_dir / f"{stem}{args.spec_suffix}"
                if not spec_path.exists():
                    print(f"[WARN] Missing spec for {stem}: {spec_path}")
                    continue

                color_parts = [int(c) for c in args.overlay_color.split(",")]
                if args.save_mask:
                    Image.fromarray((diff_mask.astype(np.uint8) * 255), mode="L").save(out_dir / f"{stem}_diff_mask.png")

                if args.overlay_from_regions:
                    if args.overlay_all_region_methods:
                        overlay_methods = list(args.region_methods)
                    else:
                        overlay_methods = [args.overlay_region_method]

                    for overlay_method in overlay_methods:
                        region_img = (
                            region_root
                            / overlay_method
                            / f"{stem}_{overlay_method}{args.overlay_region_suffix}"
                        )
                        if not region_img.exists():
                            print(f"[WARN] Missing region image for overlay ({overlay_method}): {region_img}")
                            continue
                        base_img = Image.open(region_img).convert("RGB").resize((args.img_size, args.img_size), Image.BICUBIC)
                        mask_img = resize_mask(mask, base_img.size)
                        if args.overlay_all_region_methods:
                            out_path = out_dir / f"{stem}_diff_overlay_{overlay_method}.png"
                        else:
                            out_path = out_dir / f"{stem}_diff_overlay.png"
                        if out_path.exists() and not args.overwrite:
                            print(f"[SKIP] {stem} overlay exists -> {out_path}")
                            continue
                        out_img = overlay_mask_on_image(
                            base_img, mask_img, args.alpha, color_parts, grayscale_base=args.grayscale_base
                        )
                        out_img.save(out_path)
                else:
                    base_img = Image.open(spec_path).convert("RGB").resize((args.img_size, args.img_size), Image.BICUBIC)
                    mask_img = resize_mask(mask, base_img.size)
                    out_path = out_dir / f"{stem}_diff_overlay.png"
                    if out_path.exists() and not args.overwrite:
                        print(f"[SKIP] {stem} overlay exists -> {out_path}")
                        continue
                    out_img = overlay_mask_on_image(
                        base_img, mask_img, args.alpha, color_parts, grayscale_base=args.grayscale_base
                    )
                    out_img.save(out_path)
            for method in args.region_methods:
                pth_path = region_root / method / f"{stem}_{method}_masks.pth"
                if not pth_path.exists():
                    print(f"[WARN] Missing region masks for {stem} ({method}): {pth_path}")
                    continue
                region_masks = _load_region_masks(pth_path)
                for ridx, region_mask in enumerate(region_masks, 1):
                    if args.flip_region_masks_vertical:
                        region_mask = np.flipud(region_mask)
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

        # Top-k explained-difference summary:
        # explain_ratio_topk = sum(top-k overlap_pixels) / sum(all overlap_pixels)
        by_image_method = {}
        for r in rows:
            key = (r["image"], r["method"])
            by_image_method.setdefault(key, []).append(int(r["overlap_pixels"]))

        per_image_method_rows = []
        by_method = {}
        for (image, method), overlaps in by_image_method.items():
            if len(overlaps) == 0:
                continue
            total_overlap = int(np.sum(overlaps))
            if total_overlap <= 0:
                explain_ratio = 0.0
                topk_overlap = 0
            else:
                topk_overlap = int(np.sum(sorted(overlaps, reverse=True)[: max(1, args.topk_explain_k)]))
                explain_ratio = float(topk_overlap) / float(total_overlap)
            passed = int(explain_ratio >= args.topk_explain_min_ratio)
            per_image_method_rows.append(
                {
                    "image": image,
                    "method": method,
                    "topk": int(args.topk_explain_k),
                    "topk_overlap_pixels": topk_overlap,
                    "total_overlap_pixels": total_overlap,
                    "explain_ratio_topk": explain_ratio,
                    "pass_topk_ratio": passed,
                }
            )
            by_method.setdefault(method, []).append(explain_ratio)

        summary_img_csv = out_dir / "topk_explain_ratio_per_image_method.csv"
        with open(summary_img_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "method",
                    "topk",
                    "topk_overlap_pixels",
                    "total_overlap_pixels",
                    "explain_ratio_topk",
                    "pass_topk_ratio",
                ],
            )
            writer.writeheader()
            writer.writerows(per_image_method_rows)
        print(f"[INFO] Wrote top-k explain ratio per image/method -> {summary_img_csv}")

        summary_method_rows = []
        for method, vals in sorted(by_method.items()):
            arr = np.array(vals, dtype=np.float32)
            pass_rate = float(np.mean(arr >= float(args.topk_explain_min_ratio))) if arr.size else 0.0
            summary_method_rows.append(
                {
                    "method": method,
                    "num_images": int(arr.size),
                    "mean_explain_ratio_topk": float(arr.mean()) if arr.size else 0.0,
                    "median_explain_ratio_topk": float(np.median(arr)) if arr.size else 0.0,
                    "pass_rate_topk_ratio": pass_rate,
                    "threshold": float(args.topk_explain_min_ratio),
                    "recommended": int((float(np.median(arr)) >= float(args.topk_explain_min_ratio)) if arr.size else 0),
                }
            )

        summary_method_csv = out_dir / "topk_explain_ratio_per_method.csv"
        with open(summary_method_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "num_images",
                    "mean_explain_ratio_topk",
                    "median_explain_ratio_topk",
                    "pass_rate_topk_ratio",
                    "threshold",
                    "recommended",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_method_rows)
        print(f"[INFO] Wrote top-k explain ratio per method -> {summary_method_csv}")

    print("[DONE] All pairs processed.")


if __name__ == "__main__":
    main()
