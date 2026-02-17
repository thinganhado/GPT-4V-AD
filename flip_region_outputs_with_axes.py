#!/usr/bin/env python3
import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Match defaults used in GPT-4V-AD/make_specs_768.py.
DEFAULT_SR = 16000
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 1024
DEFAULT_HOP = 256


def _load_and_flip_vertical(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        arr = np.array(rgb)
    # Bottom-to-top flip.
    return np.flipud(arr)


def _save_with_axes(
    flipped_rgb: np.ndarray,
    out_path: Path,
    duration_sec: float,
    sr: int,
    n_mels: int,
    dpi: int,
) -> None:
    h, w = flipped_rgb.shape[:2]
    fig_w = (w + 220) / dpi
    fig_h = (h + 140) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Plot using mel-bin coordinates on y; convert y tick labels to kHz.
    ax.imshow(
        flipped_rgb,
        origin="lower",
        aspect="auto",
        extent=[0.0, duration_sec, 0, n_mels - 1],
    )

    mel_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
    y_tick_bins = np.linspace(0, n_mels - 1, 6).round().astype(int)
    y_tick_labels = [f"{mel_hz[b] / 1000.0:.1f}" for b in y_tick_bins]
    x_ticks = np.linspace(0.0, duration_sec, 6)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_tick_bins)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("")

    # Keep only axes/ticks; no title.
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.98)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flip region_division spectrogram outputs vertically and add x/y axes."
    )
    parser.add_argument("--input-root", required=True, help="Root directory containing input PNG files.")
    parser.add_argument("--output-root", required=True, help="Root directory for processed PNG files.")
    parser.add_argument(
        "--glob",
        default="*_img_edge_number.png",
        help="Glob pattern under input-root (default: *_img_edge_number.png).",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=4.0,
        help="Time range shown on x-axis in seconds.",
    )
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Sample rate for y-axis scaling.")
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS, help="Number of mel bins.")
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT, help="Kept for parity with make_specs_768.py.")
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP, help="Kept for parity with make_specs_768.py.")
    parser.add_argument("--dpi", type=int, default=120, help="Output figure DPI.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.exists():
        raise SystemExit(f"--input-root does not exist: {input_root}")

    files = sorted(input_root.rglob(args.glob))
    if not files:
        raise SystemExit(f"No files matched pattern '{args.glob}' under {input_root}")

    written = 0
    skipped = 0
    for i, src in enumerate(files, start=1):
        rel = src.relative_to(input_root)
        dst = output_root / rel
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        flipped = _load_and_flip_vertical(src)
        _save_with_axes(
            flipped_rgb=flipped,
            out_path=dst,
            duration_sec=args.duration_sec,
            sr=args.sr,
            n_mels=args.n_mels,
            dpi=args.dpi,
        )
        written += 1
        if i % 100 == 0:
            print(f"[{i}/{len(files)}] written={written} skipped={skipped}")

    print(f"Done. total={len(files)} written={written} skipped={skipped} out_root={output_root}")


if __name__ == "__main__":
    main()
