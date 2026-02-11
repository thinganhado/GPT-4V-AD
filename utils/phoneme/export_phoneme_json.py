#!/usr/bin/env python3
"""
Export phoneme-level timestamps to one JSON per utterance.

Supported input formats:
- ctm, csv, tsv (single file)
- mfa_json (a single MFA JSON file or a directory containing MFA JSON files)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert phoneme timestamps to per-utterance JSON files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to alignment input: file (.ctm/.csv/.tsv/.json) or directory of MFA JSON files.",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "ctm", "csv", "tsv", "mfa_json"],
        default="auto",
        help="Input format. Auto infers from extension or directory type.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write one JSON file per utterance.",
    )
    parser.add_argument(
        "--transcript-csv",
        type=str,
        default=None,
        help="Optional CSV/TSV with utterance text. Columns: utterance_id,text.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate used for frame mapping.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="STFT hop length used for frame mapping.",
    )
    parser.add_argument(
        "--frame-rounding",
        choices=["floor", "round", "ceil"],
        default="round",
        help="How to convert time->frame index.",
    )
    parser.add_argument(
        "--drop-silence",
        action="store_true",
        default=False,
        help="Drop common silence tokens (sil/sp/spn/<eps>).",
    )
    parser.add_argument(
        "--silence-tokens",
        nargs="+",
        default=["sil", "sp", "spn", "<eps>", "SIL", "SPN"],
        help="Tokens treated as silence when --drop-silence is set.",
    )
    return parser.parse_args()


def infer_format(input_path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    if input_path.is_dir():
        return "mfa_json"
    ext = input_path.suffix.lower()
    if ext == ".ctm":
        return "ctm"
    if ext == ".csv":
        return "csv"
    if ext in [".tsv", ".txt"]:
        return "tsv"
    if ext == ".json":
        return "mfa_json"
    raise ValueError(
        f"Could not infer input format from '{input_path}'. Use --input-format explicitly."
    )


def to_frame(t_sec: float, sample_rate: int, hop_length: int, mode: str) -> int:
    v = (t_sec * sample_rate) / float(hop_length)
    if mode == "floor":
        return int(math.floor(v))
    if mode == "ceil":
        return int(math.ceil(v))
    return int(round(v))


def load_transcripts(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.excel_tab if "\t" in sample else csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"Transcript file has no header: {path}")

        columns = {c.lower(): c for c in reader.fieldnames}
        if "utterance_id" not in columns or "text" not in columns:
            raise ValueError(
                "Transcript file must contain headers 'utterance_id' and 'text'."
            )

        out: Dict[str, str] = {}
        for row in reader:
            utt = str(row[columns["utterance_id"]]).strip()
            txt = str(row[columns["text"]]).strip()
            if utt:
                out[utt] = txt
        return out


def read_ctm(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid CTM line {line_no}: '{line}'")
            utt = parts[0]
            start = float(parts[2])
            dur = float(parts[3])
            label = parts[4]
            conf = float(parts[5]) if len(parts) > 5 else None
            rows.append(
                {
                    "utterance_id": utt,
                    "phoneme": label,
                    "start_sec": start,
                    "end_sec": start + dur,
                    "duration_sec": dur,
                    "confidence": conf,
                }
            )
    return rows


def read_delimited(path: Path, fmt: str) -> List[dict]:
    delimiter = "," if fmt == "csv" else "\t"
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Input file has no header: {path}")

        cols = {c.lower(): c for c in reader.fieldnames}
        required = ["utterance_id", "phoneme", "start_sec"]
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(
                f"Missing required columns {missing}. "
                "Need utterance_id, phoneme, start_sec, and one of end_sec/duration_sec."
            )
        has_end = "end_sec" in cols
        has_dur = "duration_sec" in cols
        if not has_end and not has_dur:
            raise ValueError("Need either 'end_sec' or 'duration_sec' column.")

        for row in reader:
            utt = str(row[cols["utterance_id"]]).strip()
            ph = str(row[cols["phoneme"]]).strip()
            start = float(row[cols["start_sec"]])
            if has_end:
                end = float(row[cols["end_sec"]])
                dur = max(0.0, end - start)
            else:
                dur = float(row[cols["duration_sec"]])
                end = start + dur
            conf = None
            if "confidence" in cols and row[cols["confidence"]] not in (None, ""):
                conf = float(row[cols["confidence"]])
            rows.append(
                {
                    "utterance_id": utt,
                    "phoneme": ph,
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": dur,
                    "confidence": conf,
                }
            )
    return rows


def _coerce_entry(entry) -> Optional[Tuple[float, float, str]]:
    if isinstance(entry, dict):
        start = entry.get("begin", entry.get("start", entry.get("start_sec")))
        end = entry.get("end", entry.get("end_sec"))
        label = entry.get("label", entry.get("text", entry.get("phone", entry.get("phoneme"))))
    elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
        start, end, label = entry[0], entry[1], entry[2]
    else:
        return None

    if start is None or end is None or label is None:
        return None
    try:
        start_f = float(start)
        end_f = float(end)
    except Exception:
        return None
    label_s = str(label).strip()
    if not label_s:
        return None
    return start_f, end_f, label_s


def _extract_phone_entries(obj) -> List[Tuple[float, float, str]]:
    # Common MFA JSON shape: {"tiers": {"phones": {"entries": [...]}}}
    if isinstance(obj, dict):
        tiers = obj.get("tiers")
        if isinstance(tiers, dict):
            for key in ["phones", "phone", "phoneme", "phonemes"]:
                tier = tiers.get(key)
                if isinstance(tier, dict) and isinstance(tier.get("entries"), list):
                    out = []
                    for e in tier["entries"]:
                        c = _coerce_entry(e)
                        if c is not None:
                            out.append(c)
                    if out:
                        return out

        # Alternative shapes: top-level list-like phone containers
        for key in ["phones", "phonemes", "segments", "entries"]:
            maybe = obj.get(key)
            if isinstance(maybe, list):
                out = []
                for e in maybe:
                    c = _coerce_entry(e)
                    if c is not None:
                        out.append(c)
                if out:
                    return out

    # Entire object is already a list of entries
    if isinstance(obj, list):
        out = []
        for e in obj:
            c = _coerce_entry(e)
            if c is not None:
                out.append(c)
        if out:
            return out

    return []


def read_mfa_json(input_path: Path) -> List[dict]:
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob("*.json"))

    rows: List[dict] = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        utt = file_path.stem
        entries = _extract_phone_entries(obj)
        if not entries:
            continue

        for start, end, label in entries:
            rows.append(
                {
                    "utterance_id": utt,
                    "phoneme": label,
                    "start_sec": float(start),
                    "end_sec": float(end),
                    "duration_sec": max(0.0, float(end) - float(start)),
                    "confidence": None,
                }
            )
    return rows


def group_rows(rows: List[dict], drop_silence: bool, silence_tokens: List[str]) -> Dict[str, List[dict]]:
    silence_set = set(silence_tokens)
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        utt = row["utterance_id"]
        ph = row["phoneme"]
        if drop_silence and ph in silence_set:
            continue
        grouped.setdefault(utt, []).append(row)
    for utt in grouped:
        grouped[utt].sort(key=lambda x: (x["start_sec"], x["end_sec"]))
    return grouped


def build_utterance_json(
    utterance_id: str,
    phonemes: List[dict],
    transcript: Optional[str],
    sample_rate: int,
    hop_length: int,
    frame_rounding: str,
) -> dict:
    items = []
    for i, p in enumerate(phonemes):
        start_sec = float(p["start_sec"])
        end_sec = float(p["end_sec"])
        items.append(
            {
                "index": i,
                "phoneme": p["phoneme"],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": max(0.0, end_sec - start_sec),
                "start_frame": to_frame(start_sec, sample_rate, hop_length, frame_rounding),
                "end_frame": to_frame(end_sec, sample_rate, hop_length, frame_rounding),
                "confidence": p.get("confidence"),
            }
        )

    if items:
        start = items[0]["start_sec"]
        end = items[-1]["end_sec"]
    else:
        start, end = 0.0, 0.0

    return {
        "utterance_id": utterance_id,
        "transcript": transcript,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "frame_rounding": frame_rounding,
        "num_phonemes": len(items),
        "start_sec": start,
        "end_sec": end,
        "duration_sec": max(0.0, end - start),
        "phonemes": items,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    fmt = infer_format(input_path, args.input_format)
    if fmt == "ctm":
        rows = read_ctm(input_path)
    elif fmt in {"csv", "tsv"}:
        rows = read_delimited(input_path, fmt)
    elif fmt == "mfa_json":
        rows = read_mfa_json(input_path)
    else:
        raise ValueError(f"Unsupported input format: {fmt}")

    transcripts = load_transcripts(Path(args.transcript_csv)) if args.transcript_csv else {}
    grouped = group_rows(rows, args.drop_silence, args.silence_tokens)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for utt, phonemes in grouped.items():
        obj = build_utterance_json(
            utterance_id=utt,
            phonemes=phonemes,
            transcript=transcripts.get(utt),
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            frame_rounding=args.frame_rounding,
        )
        out_path = out_dir / f"{utt}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=True, indent=2)

    print(f"Input format: {fmt}")
    print(f"Utterances written: {len(grouped)}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
