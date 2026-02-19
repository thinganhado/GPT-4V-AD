#!/usr/bin/env python3
"""Build fake-audio .lab transcripts for WakeFake-style pair CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map fake_path entries from a pairs CSV to transcripts from LJSpeech metadata "
            "and write one .lab per fake utterance."
        )
    )
    parser.add_argument("--pairs-csv", type=str, required=True, help="Path to pairs_wakefake.csv")
    parser.add_argument(
        "--metadata-csv",
        type=str,
        required=True,
        help="Path to LJSpeech metadata.csv (pipe-delimited).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where <fake_stem>.lab files are written.",
    )
    parser.add_argument(
        "--manifest-out",
        type=str,
        default=None,
        help="Optional TSV manifest with mapping status per row.",
    )
    parser.add_argument(
        "--pairs-real-col",
        type=str,
        default="real_path",
        help="Column name in pairs CSV for real audio path.",
    )
    parser.add_argument(
        "--pairs-fake-col",
        type=str,
        default="fake_path",
        help="Column name in pairs CSV for fake audio path.",
    )
    parser.add_argument(
        "--split-col",
        type=str,
        default="split",
        help="Split column name in pairs CSV.",
    )
    parser.add_argument(
        "--split-value",
        type=str,
        default=None,
        help="Optional split filter (e.g., test). If unset, uses all rows.",
    )
    parser.add_argument(
        "--metadata-id-col-index",
        type=int,
        default=0,
        help="0-based column index for utterance ID in metadata.csv.",
    )
    parser.add_argument(
        "--metadata-text-col-index",
        type=int,
        default=2,
        help="0-based column index for transcript text in metadata.csv (default: 2 = column 3).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Do not overwrite existing .lab files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit non-zero if any rows cannot be resolved.",
    )
    return parser.parse_args()


def norm(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_ljspeech_metadata(metadata_csv: Path, id_idx: int, text_idx: int) -> Dict[str, str]:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    out: Dict[str, str] = {}
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            max_idx = max(id_idx, text_idx)
            if len(row) <= max_idx:
                continue
            utt_id = norm(row[id_idx])
            text = norm(row[text_idx])
            if not utt_id or not text:
                continue
            out[utt_id] = text

    if not out:
        raise ValueError(f"No usable metadata rows loaded from: {metadata_csv}")
    return out


def main() -> None:
    args = parse_args()

    pairs_csv = Path(args.pairs_csv)
    metadata_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)

    if not pairs_csv.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")

    transcript_by_real_id = load_ljspeech_metadata(
        metadata_csv=metadata_csv,
        id_idx=args.metadata_id_col_index,
        text_idx=args.metadata_text_col_index,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    rows_out: List[dict] = []
    total = 0
    filtered = 0
    written = 0
    skipped_existing = 0
    missing_real_id = 0
    missing_transcript = 0
    invalid_fake_path = 0

    with open(pairs_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Pairs CSV has no header: {pairs_csv}")

        required = [args.pairs_real_col, args.pairs_fake_col]
        missing_cols = [c for c in required if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Pairs CSV missing required columns: {missing_cols}")

        for row in reader:
            total += 1
            split = norm(row.get(args.split_col, "")) if args.split_col else ""
            if args.split_value and split != args.split_value:
                filtered += 1
                continue

            real_path_s = norm(row.get(args.pairs_real_col, ""))
            fake_path_s = norm(row.get(args.pairs_fake_col, ""))
            real_id = Path(real_path_s).stem if real_path_s else ""
            fake_stem = Path(fake_path_s).stem if fake_path_s else ""

            status = "written"
            transcript = ""

            if not fake_stem:
                status = "invalid_fake_path"
                invalid_fake_path += 1
            elif not real_id:
                status = "missing_real_id"
                missing_real_id += 1
            else:
                transcript = transcript_by_real_id.get(real_id, "")
                if not transcript:
                    status = "missing_transcript"
                    missing_transcript += 1
                else:
                    lab_path = output_dir / f"{fake_stem}.lab"
                    if lab_path.exists() and args.skip_existing:
                        status = "skipped_existing"
                        skipped_existing += 1
                    else:
                        with open(lab_path, "w", encoding="utf-8") as wf:
                            wf.write(transcript + "\n")
                        written += 1

            rows_out.append(
                {
                    "real_id": real_id,
                    "fake_id": fake_stem,
                    "real_path": real_path_s,
                    "fake_path": fake_path_s,
                    "split": split,
                    "status": status,
                    "transcript": transcript,
                }
            )

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8", newline="") as mf:
            writer = csv.DictWriter(
                mf,
                delimiter="\t",
                fieldnames=[
                    "real_id",
                    "fake_id",
                    "real_path",
                    "fake_path",
                    "split",
                    "status",
                    "transcript",
                ],
            )
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"manifest_written={manifest_path}")

    print(f"pairs_rows_total={total}")
    if args.split_value:
        print(f"rows_filtered_by_split={filtered}")
    print(f"lab_written={written}")
    print(f"skipped_existing={skipped_existing}")
    print(f"missing_real_id={missing_real_id}")
    print(f"missing_transcript={missing_transcript}")
    print(f"invalid_fake_path={invalid_fake_path}")
    print(f"output_dir={output_dir}")

    unresolved = missing_real_id + missing_transcript + invalid_fake_path
    if args.strict and unresolved > 0:
        raise SystemExit("Strict mode enabled and unresolved rows were found.")


if __name__ == "__main__":
    main()
