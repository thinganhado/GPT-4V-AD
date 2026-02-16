#!/usr/bin/env python3
"""
Select region-division sizing candidates from region overlap stats.

Input:
  - CSV from make_masks_overlay.py: region_diff_stats.csv
    columns: image, method, region_id, overlap_pixels, ...

Logic:
  explain_ratio_topk(image, method) =
      sum(top-k overlap_pixels) / sum(all overlap_pixels)

Outputs:
  - per-image/per-method explain ratios
  - per-method summary with pass rates for threshold
  - printed recommended sizing args for region_division.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select grid/superpixel sizing from overlap stats.")
    p.add_argument("--region-diff-csv", required=True, help="Path to region_diff_stats.csv.")
    p.add_argument("--topk", type=int, default=3, help="Top-k regions used in explain ratio.")
    p.add_argument("--min-ratio", type=float, default=0.5, help="Target explained ratio threshold.")
    p.add_argument("--output-dir", default=None, help="Optional output directory for summaries.")
    p.add_argument(
        "--selection-mode",
        choices=["best_median", "max_granularity_pass"],
        default="max_granularity_pass",
        help=(
            "best_median: choose candidate with highest median ratio. "
            "max_granularity_pass: choose most fine-grained candidate that still meets threshold."
        ),
    )
    return p.parse_args()


def parse_method(method: str) -> tuple[str, int | None, float | None]:
    if method.startswith("grid_n"):
        m = re.match(r"^grid_n(\d+)$", method)
        if m:
            return "grid", int(m.group(1)), None
    if method == "grid":
        return "grid", None, None

    m = re.match(r"^superpixel_n(\d+)_c([0-9.]+)$", method)
    if m:
        return "superpixel", int(m.group(1)), float(m.group(2))
    if method == "superpixel":
        return "superpixel", None, None

    if method.startswith("sam"):
        return "sam", None, None
    return method, None, None


def main() -> None:
    args = parse_args()
    in_csv = Path(args.region_diff_csv).expanduser().resolve()
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    rows = []
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"image", "method", "region_id", "overlap_pixels"}
        if r.fieldnames is None or not required.issubset(set(r.fieldnames)):
            raise ValueError(f"CSV must include columns: {sorted(required)}")
        for row in r:
            rows.append(
                {
                    "image": str(row["image"]).strip(),
                    "method": str(row["method"]).strip(),
                    "region_id": int(row["region_id"]),
                    "overlap_pixels": int(float(row["overlap_pixels"])),
                }
            )

    by_img_method: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        by_img_method[(row["image"], row["method"])].append(row["overlap_pixels"])

    per_image_method = []
    by_method_ratio: dict[str, list[float]] = defaultdict(list)
    for (image, method), overlaps in by_img_method.items():
        total = sum(overlaps)
        topk_sum = sum(sorted(overlaps, reverse=True)[: max(1, args.topk)])
        ratio = (topk_sum / total) if total > 0 else 0.0
        passed = 1 if ratio >= args.min_ratio else 0
        per_image_method.append(
            {
                "image": image,
                "method": method,
                "topk": args.topk,
                "topk_overlap_pixels": topk_sum,
                "total_overlap_pixels": total,
                "explain_ratio_topk": ratio,
                "pass_topk_ratio": passed,
            }
        )
        by_method_ratio[method].append(ratio)

    method_summary = []
    for method in sorted(by_method_ratio):
        vals = by_method_ratio[method]
        pass_rate = sum(v >= args.min_ratio for v in vals) / len(vals) if vals else 0.0
        method_summary.append(
            {
                "method": method,
                "num_images": len(vals),
                "mean_explain_ratio_topk": mean(vals) if vals else 0.0,
                "median_explain_ratio_topk": median(vals) if vals else 0.0,
                "pass_rate_topk_ratio": pass_rate,
                "threshold": args.min_ratio,
                "recommended": 1 if (median(vals) >= args.min_ratio if vals else False) else 0,
            }
        )

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else in_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    per_img_csv = out_dir / "topk_explain_ratio_per_image_method.csv"
    per_m_csv = out_dir / "topk_explain_ratio_per_method.csv"

    with per_img_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
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
        w.writeheader()
        w.writerows(per_image_method)

    with per_m_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
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
        w.writeheader()
        w.writerows(method_summary)

    # Recommend candidate per base method.
    grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    for row in method_summary:
        base, nseg, comp = parse_method(row["method"])
        grouped_candidates[base].append(
            {
                "method": row["method"],
                "median": float(row["median_explain_ratio_topk"]),
                "pass_rate": float(row["pass_rate_topk_ratio"]),
                "nseg": nseg,
                "compactness": comp,
            }
        )

    print(f"[INFO] per-image summary: {per_img_csv}")
    print(f"[INFO] per-method summary: {per_m_csv}")
    print(f"[INFO] target: top{args.topk} >= {args.min_ratio:.3f}")

    for base in sorted(grouped_candidates):
        cands = grouped_candidates[base]
        # always keep a fallback by best median
        best_median = sorted(cands, key=lambda x: (x["median"], x["pass_rate"]), reverse=True)[0]

        if args.selection_mode == "best_median":
            chosen = best_median
            reason = "best_median"
        else:
            passing = [c for c in cands if c["median"] >= args.min_ratio]
            if base in {"grid", "superpixel"} and passing:
                # choose most fine-grained candidate (largest n), break ties by median/pass rate
                chosen = sorted(
                    passing,
                    key=lambda x: (x["nseg"] if x["nseg"] is not None else -1, x["median"], x["pass_rate"]),
                    reverse=True,
                )[0]
                reason = "max_granularity_pass"
            elif passing:
                chosen = sorted(passing, key=lambda x: (x["median"], x["pass_rate"]), reverse=True)[0]
                reason = "max_granularity_pass"
            else:
                chosen = best_median
                reason = "fallback_best_median_no_pass"

        print(
            f"[BEST] {base}: method={chosen['method']} "
            f"median={chosen['median']:.4f} pass_rate={chosen['pass_rate']:.4f} "
            f"(mode={reason})"
        )
        if base == "grid":
            if chosen["nseg"] is not None:
                print(f"  region_division.py arg: --div_num {chosen['nseg']}")
            else:
                print("  region_division.py arg: --div_num <current grid default>")
        if base == "superpixel":
            if chosen["nseg"] is not None and chosen["compactness"] is not None:
                print(
                    f"  region_division.py args: --slic-n-segments {chosen['nseg']} "
                    f"--slic-compactness {chosen['compactness']}"
                )
            else:
                print("  region_division.py args: --slic-n-segments <default> --slic-compactness <default>")


if __name__ == "__main__":
    main()
