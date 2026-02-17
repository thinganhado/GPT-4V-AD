#!/usr/bin/env python3
import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Match defaults from GPT-4V-AD/make_specs_768.py
DEFAULT_SR = 16000
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 1024
DEFAULT_HOP = 256

DEFAULT_INPUT_ROOT = "/scratch3/che489/Ha/interspeech/localization/Ms_region_outputs"
DEFAULT_OUTPUT_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_specs"


def _load_and_flip_vertical(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        arr = np.array(img.convert("RGB"))
    # Bottom-to-top
    return arr


def _save_with_axes(
    flipped_rgb: np.ndarray,
    out_path: Path,
    duration_sec: float,
    sr: int,
    n_mels: int,
    dpi: int,
) -> None:
    h, w = flipped_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=((w + 220) / dpi, (h + 140) / dpi), dpi=dpi)

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
    ax.set_title("")  # no title

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _resolve_methods(input_root: Path, methods_arg: str) -> list[Path]:
    if methods_arg.strip().lower() == "all":
        return sorted([p for p in input_root.iterdir() if p.is_dir()])

    wanted = {m.strip().lower() for m in methods_arg.split(",") if m.strip()}
    out = []
    for name in sorted(wanted):
        p = input_root / name
        if not p.is_dir():
            raise SystemExit(f"Method folder not found under input-root: {p}")
        out.append(p)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Flip Ms_region_outputs images vertically and add x/y axes (no title)."
    )
    p.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--methods",
        default="all",
        help='Comma-separated methods (e.g., "grid,sam,superpixel") or "all".',
    )
    p.add_argument("--glob", default="*_img_edge_number.png")
    p.add_argument("--duration-sec", type=float, default=4.0)
    p.add_argument("--sr", type=int, default=DEFAULT_SR)
    p.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    p.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)  # parity with make_specs_768.py
    p.add_argument("--hop", type=int, default=DEFAULT_HOP)  # parity with make_specs_768.py
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"--input-root does not exist: {input_root}")

    method_dirs = _resolve_methods(input_root, args.methods)
    if not method_dirs:
        raise SystemExit(f"No method folders found under: {input_root}")

    total = 0
    written = 0
    skipped = 0
    for method_dir in method_dirs:
        method = method_dir.name
        files = sorted(method_dir.rglob(args.glob))
        if not files:
            print(f"[skip] no matches for method={method}")
            continue

        for src in files:
            total += 1
            rel_under_method = src.relative_to(method_dir)
            dst = output_root / method / rel_under_method
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
            if total % 100 == 0:
                print(f"[{total}] written={written} skipped={skipped}")

    print(f"Done. total={total} written={written} skipped={skipped} out_root={output_root}")


if __name__ == "__main__":
    main()

