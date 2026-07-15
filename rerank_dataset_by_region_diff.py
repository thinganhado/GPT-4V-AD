#!/usr/bin/env python3
"""Rerank each sample's region rows using region_diff_stats.csv."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort region rows within each sample by reconstructed artifact-overlap score."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--region-diff-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--audit-csv", default="", help="Optional rank/score audit CSV.")
    parser.add_argument("--method", default="grid")
    parser.add_argument(
        "--score-column",
        choices=("overlap_pixels", "coverage"),
        default="overlap_pixels",
    )
    parser.add_argument("--expected-regions-per-sample", type=int, default=3)
    return parser.parse_args()


def normalize_sample_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).stem


def load_scores(path: Path, method: str, score_column: str) -> Dict[Tuple[str, int], float]:
    scores: Dict[Tuple[str, int], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"method", "region_id", score_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Region-diff CSV is missing columns: {sorted(missing)}")
        if not ({"image", "sample_id"} & set(reader.fieldnames or [])):
            raise ValueError("Region-diff CSV must contain either 'image' or 'sample_id'.")

        for row in reader:
            if str(row.get("method", "")).strip() != method:
                continue
            sample_id = normalize_sample_id(row.get("sample_id") or row.get("image"))
            try:
                region_id = int(str(row.get("region_id", "")).strip())
                score = float(str(row.get(score_column, "")).strip())
            except (TypeError, ValueError):
                continue
            key = (sample_id, region_id)
            if key in scores and scores[key] != score:
                raise ValueError(f"Conflicting duplicate score for {key}: {scores[key]} vs {score}")
            scores[key] = score

    if not scores:
        raise ValueError(f"No scores loaded for method={method!r} from {path}")
    return scores


def main() -> None:
    args = parse_args()
    source = Path(args.input_csv).expanduser().resolve()
    stats = Path(args.region_diff_csv).expanduser().resolve()
    destination = Path(args.output_csv).expanduser().resolve()
    audit_path = Path(args.audit_csv).expanduser().resolve() if args.audit_csv else None
    if source == destination:
        raise ValueError("Input and output must differ so the source ordering remains auditable.")

    scores = load_scores(stats, args.method, args.score_column)
    grouped: "OrderedDict[str, List[dict[str, str]]]" = OrderedDict()
    with source.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        required = {"sample_id", "region_id"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
        for row in reader:
            sample_id = normalize_sample_id(row.get("sample_id"))
            grouped.setdefault(sample_id, []).append(row)

    output_rows: List[dict[str, str]] = []
    audit_rows: List[dict[str, object]] = []
    missing_scores: List[Tuple[str, int]] = []
    bad_group_sizes: List[Tuple[str, int]] = []
    changed_orders = 0

    for sample_id, rows in grouped.items():
        if args.expected_regions_per_sample > 0 and len(rows) != args.expected_regions_per_sample:
            bad_group_sizes.append((sample_id, len(rows)))
            continue
        scored = []
        for original_position, row in enumerate(rows, start=1):
            try:
                region_id = int(str(row.get("region_id", "")).strip())
            except ValueError:
                raise ValueError(f"Invalid region_id for sample {sample_id}: {row.get('region_id')!r}")
            key = (sample_id, region_id)
            if key not in scores:
                missing_scores.append(key)
                continue
            scored.append((row, region_id, scores[key], original_position))

        if len(scored) != len(rows):
            continue
        ranked = sorted(scored, key=lambda item: (-item[2], item[1]))
        original_ids = [item[1] for item in scored]
        ranked_ids = [item[1] for item in ranked]
        if ranked_ids != original_ids:
            changed_orders += 1
        for rank, (row, region_id, score, original_position) in enumerate(ranked, start=1):
            output_rows.append(row)
            audit_rows.append(
                {
                    "sample_id": sample_id,
                    "rank": rank,
                    "region_id": region_id,
                    "score_column": args.score_column,
                    "score": score,
                    "original_position": original_position,
                }
            )

    if bad_group_sizes:
        raise ValueError(
            f"Found {len(bad_group_sizes)} samples with an unexpected row count; "
            f"examples={bad_group_sizes[:10]}"
        )
    if missing_scores:
        raise ValueError(
            f"Missing reconstructed scores for {len(missing_scores)} sample/region pairs; "
            f"examples={missing_scores[:10]}"
        )
    if len(output_rows) != sum(len(rows) for rows in grouped.values()):
        raise ValueError("Output row count differs from input row count.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    if audit_path:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8", newline="") as f:
            fields = ["sample_id", "rank", "region_id", "score_column", "score", "original_position"]
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(audit_rows)

    print(f"samples={len(grouped)}")
    print(f"rows={len(output_rows)}")
    print(f"changed_sample_orders={changed_orders}")
    print(f"score_column={args.score_column}")
    print(f"output_csv={destination}")
    if audit_path:
        print(f"audit_csv={audit_path}")


if __name__ == "__main__":
    main()
