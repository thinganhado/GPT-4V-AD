#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib.cm as cm
import numpy as np
from PIL import Image


def audio_to_spec_png(audio_path: Path, out_path: Path, sr: int, n_mels: int, n_fft: int, hop: int):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=60.0)
    mel_norm = (mel_db + 60.0) / 60.0
    mel_color = cm.get_cmap("magma")(mel_norm)
    mel_img = (mel_color[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(mel_img, mode="RGB").resize((768, 768), Image.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def collect_from_in_dir(in_dir: Path) -> List[Tuple[Path, Path]]:
    audios = sorted(list(in_dir.rglob("*.wav")) + list(in_dir.rglob("*.flac")))
    if not audios:
        raise SystemExit(f"No .wav/.flac files found under {in_dir}")
    pairs: List[Tuple[Path, Path]] = []
    for a in audios:
        rel = a.relative_to(in_dir).with_suffix(".png")
        pairs.append((a, rel))
    return pairs


def collect_from_pairs_csv(pairs_csv: Path, audio_col: str) -> List[Tuple[Path, Path]]:
    seen = set()
    pairs: List[Tuple[Path, Path]] = []
    with open(pairs_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or audio_col not in r.fieldnames:
            raise SystemExit(f"CSV must contain column '{audio_col}'")
        for row in r:
            ap = Path(str(row[audio_col]).strip())
            if not str(ap):
                continue
            # Deduplicate by absolute path so repeated real_path entries are processed once
            key = str(ap)
            if key in seen:
                continue
            seen.add(key)
            out_rel = Path(f"{ap.stem}.png")
            pairs.append((ap, out_rel))
    if not pairs:
        raise SystemExit(f"No valid audio paths found in column '{audio_col}' from {pairs_csv}")
    return pairs


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--in-dir", type=str, help="Input root scanned recursively for .wav/.flac")
    src.add_argument("--pairs-csv", type=str, help="CSV file (e.g., pairs_vocv4.csv)")

    p.add_argument("--pairs-audio-col", type=str, default="real_path", help="Audio path column when using --pairs-csv")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)

    p.add_argument("--overwrite", action="store_true", default=False, help="Regenerate outputs even if PNG exists")
    p.add_argument("--shard-id", type=int, default=0, help="Shard index for parallel workers")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of shards/workers")
    p.add_argument("--log-every", type=int, default=100, help="Print progress every N items")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    if args.in_dir:
        all_pairs = collect_from_in_dir(Path(args.in_dir))
    else:
        all_pairs = collect_from_pairs_csv(Path(args.pairs_csv), args.pairs_audio_col)

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit("--shard-id must be in [0, num_shards)")

    pairs = all_pairs[args.shard_id::args.num_shards]
    total = len(pairs)
    written = 0
    skipped_existing = 0
    missing_audio = 0

    print(
        f"source_total={len(all_pairs)} shard_total={total} "
        f"shard_id={args.shard_id}/{args.num_shards} out_dir={out_dir}"
    )

    for i, (audio_path, out_rel) in enumerate(pairs, 1):
        out_path = out_dir / out_rel

        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            if args.log_every > 0 and i % args.log_every == 0:
                print(
                    f"[{i}/{total}] written={written} skipped_existing={skipped_existing} "
                    f"missing_audio={missing_audio}"
                )
            continue

        if not audio_path.exists():
            missing_audio += 1
            if args.log_every > 0 and i % args.log_every == 0:
                print(
                    f"[{i}/{total}] written={written} skipped_existing={skipped_existing} "
                    f"missing_audio={missing_audio}"
                )
            continue

        if args.log_every == 1:
            print(f"[{i}/{total}] generating {out_rel}")
        audio_to_spec_png(audio_path, out_path, args.sr, args.n_mels, args.n_fft, args.hop)
        written += 1

        if args.log_every > 1 and i % args.log_every == 0:
            print(
                f"[{i}/{total}] written={written} skipped_existing={skipped_existing} "
                f"missing_audio={missing_audio}"
            )

    print(
        f"Done. shard_total={total} written={written} skipped_existing={skipped_existing} "
        f"missing_audio={missing_audio} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
