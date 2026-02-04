#!/usr/bin/env python3
import argparse
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
import matplotlib.cm as cm


def wav_to_spec_png(wav_path: Path, out_path: Path, sr: int, n_mels: int, n_fft: int, hop: int):
    y, _sr = librosa.load(wav_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=60.0)
    mel_norm = (mel_db + 60.0) / 60.0
    mel_color = cm.get_cmap("magma")(mel_norm)
    mel_img = (mel_color[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(mel_img, mode="RGB").resize((768, 768), Image.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    wavs = sorted(in_dir.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"No wav files found under {in_dir}")

    for wav in wavs:
        rel = wav.relative_to(in_dir).with_suffix(".png")
        out_path = out_dir / rel
        wav_to_spec_png(wav, out_path, args.sr, args.n_mels, args.n_fft, args.hop)

    print(f"Saved {len(wavs)} spectrograms to {out_dir}")


if __name__ == "__main__":
    main()
