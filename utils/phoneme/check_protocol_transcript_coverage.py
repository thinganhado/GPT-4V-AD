#!/usr/bin/env python3
"""Check transcript coverage for vocoder-prefixed protocol entries."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare protocol audio IDs (e.g., vocoder_LA_T_xxx) against transcript .lab files "
            "using canonical LA_* stems."
        )
    )
    p.add_argument("--protocol", type=str, required=True, help="Path to protocol.txt")
    p.add_argument("--transcript-dir", type=str, required=True, help="Directory containing *.lab")
    p.add_argument(
        "--anchor",
        type=str,
        default="_LA_",
        help="Anchor token separating vocoder prefix from canonical stem (default: _LA_).",
    )
    p.add_argument(
        "--missing-out",
        type=str,
        default=None,
        help="Optional output txt with missing canonical stems (one per line).",
    )
    p.add_argument(
        "--show-missing",
        type=int,
        default=30,
        help="How many missing stems to print in console.",
    )
    return p.parse_args()


def canonicalize(audio_id: str, anchor: str) -> str:
    if anchor in audio_id:
        return "LA_" + audio_id.split(anchor, 1)[1]
    return audio_id


def main() -> None:
    args = parse_args()
    protocol = Path(args.protocol)
    transcript_dir = Path(args.transcript_dir)

    if not protocol.exists():
        raise FileNotFoundError(f"Protocol not found: {protocol}")
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    transcript_stems = {p.stem for p in transcript_dir.glob("*.lab")}

    all_audio_ids = []
    canonical_stems = []

    with open(protocol, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            audio_id = parts[1]
            all_audio_ids.append(audio_id)
            canonical_stems.append(canonicalize(audio_id, args.anchor))

    unique_stems = sorted(set(canonical_stems))
    missing = sorted([s for s in unique_stems if s not in transcript_stems])

    line_level_missing = sum(1 for s in canonical_stems if s not in transcript_stems)

    print(f"protocol_lines={len(all_audio_ids)}")
    print(f"unique_canonical_stems={len(unique_stems)}")
    print(f"transcript_files={len(transcript_stems)}")
    print(f"present_unique_stems={len(unique_stems) - len(missing)}")
    print(f"missing_unique_stems={len(missing)}")
    print(f"missing_line_level={line_level_missing}")

    voc_total = {}
    voc_missing = {}
    for audio_id, stem in zip(all_audio_ids, canonical_stems):
        voc = audio_id.split("_", 1)[0] if "_" in audio_id else audio_id
        voc_total[voc] = voc_total.get(voc, 0) + 1
        if stem not in transcript_stems:
            voc_missing[voc] = voc_missing.get(voc, 0) + 1

    print("vocoder_breakdown:")
    for v in sorted(voc_total):
        print(f"  {v}: total={voc_total[v]}, missing={voc_missing.get(v, 0)}")

    if missing:
        n = max(0, int(args.show_missing))
        if n > 0:
            print("missing_examples:")
            for s in missing[:n]:
                print(f"  {s}")

    if args.missing_out:
        out = Path(args.missing_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as wf:
            for s in missing:
                wf.write(s + "\n")
        print(f"missing_written={out}")


if __name__ == "__main__":
    main()
