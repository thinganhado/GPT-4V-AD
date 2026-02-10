#!/usr/bin/env python3
"""Build per-utterance .lab transcripts from ASVspoof2019 VCTK metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract transcripts from ASVspoof2019_LA_VCTK_MetaInfo.tsv and write "
            "one .lab file per ASVspoof_ID."
        )
    )
    parser.add_argument("--metadata-tsv", type=str, required=True, help="Path to metadata TSV.")
    parser.add_argument(
        "--txt-root",
        type=str,
        required=True,
        help="Root folder for referenced text files (e.g., /.../datasets/txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where <ASVspoof_ID>.lab files will be written.",
    )
    parser.add_argument(
        "--manifest-out",
        type=str,
        default=None,
        help="Optional output TSV summary path.",
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
        help="Fail if any transcript cannot be resolved.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def first_non_empty_line(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = normalize_text(line)
            if t:
                return t
    return None


def resolve_from_reference(ref_id: str, txt_root: Path) -> Tuple[Optional[str], Optional[Path]]:
    ref = ref_id.strip()
    if not ref or ref == "-":
        return None, None

    candidates: List[Path] = []
    if "/" in ref or "\\" in ref:
        p = Path(ref)
        if p.suffix:
            candidates.append(txt_root / p)
        else:
            candidates.append(txt_root / f"{ref}.txt")
    else:
        speaker = ref.split("_", 1)[0]
        candidates.append(txt_root / speaker / f"{ref}.txt")
        candidates.append(txt_root / f"{ref}.txt")

    for c in candidates:
        text = first_non_empty_line(c)
        if text is not None:
            return text, c
    return None, None


def main() -> None:
    args = parse_args()

    metadata_tsv = Path(args.metadata_tsv)
    txt_root = Path(args.txt_root)
    output_dir = Path(args.output_dir)

    if not metadata_tsv.exists():
        raise FileNotFoundError(f"Metadata TSV not found: {metadata_tsv}")
    if not txt_root.exists():
        raise FileNotFoundError(f"txt root not found: {txt_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {
        "ASVspoof_ID",
        "TTS_text",
        "VC_source_VCTK_ID",
    }

    rows_out: List[Dict[str, str]] = []
    missing: List[Tuple[str, str, str]] = []
    written = 0
    skipped = 0

    with open(metadata_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Metadata TSV has no header")

        missing_cols = [c for c in required_cols if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Metadata TSV missing required columns: {missing_cols}")

        for row in reader:
            asv_id = normalize_text(str(row.get("ASVspoof_ID", "")))
            tts_text = normalize_text(str(row.get("TTS_text", "")))
            vc_ref = normalize_text(str(row.get("VC_source_VCTK_ID", "")))
            vctk_id = normalize_text(str(row.get("VCTK_ID", "")))

            if not asv_id:
                continue

            transcript = None
            source = None

            if tts_text and tts_text != "-":
                transcript = tts_text
                source = "TTS_text"
            else:
                ref_text, ref_path = resolve_from_reference(vc_ref, txt_root)
                if ref_text is not None:
                    transcript = ref_text
                    source = str(ref_path)

            if transcript is None and vctk_id and vctk_id != "-":
                ref_text, ref_path = resolve_from_reference(vctk_id, txt_root)
                if ref_text is not None:
                    transcript = ref_text
                    source = str(ref_path)

            if transcript is None:
                missing.append((asv_id, vc_ref, vctk_id))
                rows_out.append(
                    {
                        "ASVspoof_ID": asv_id,
                        "status": "missing",
                        "source": "",
                        "transcript": "",
                    }
                )
                continue

            lab_path = output_dir / f"{asv_id}.lab"
            if lab_path.exists() and args.skip_existing:
                skipped += 1
                status = "skipped_existing"
            else:
                with open(lab_path, "w", encoding="utf-8") as wf:
                    wf.write(transcript + "\n")
                written += 1
                status = "written"

            rows_out.append(
                {
                    "ASVspoof_ID": asv_id,
                    "status": status,
                    "source": source or "",
                    "transcript": transcript,
                }
            )

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8", newline="") as mf:
            w = csv.DictWriter(
                mf,
                delimiter="\t",
                fieldnames=["ASVspoof_ID", "status", "source", "transcript"],
            )
            w.writeheader()
            w.writerows(rows_out)

    print(f"Metadata rows processed: {len(rows_out)}")
    print(f".lab written: {written}")
    print(f".lab skipped existing: {skipped}")
    print(f"Missing transcripts: {len(missing)}")
    if missing:
        print("Examples of missing (ASVspoof_ID, VC_source_VCTK_ID):")
        for asv_id, vc_ref in missing[:10]:
            print(f"  {asv_id}\t{vc_ref}")

    if args.strict and missing:
        raise SystemExit("Strict mode enabled and unresolved transcripts were found.")


if __name__ == "__main__":
    main()
