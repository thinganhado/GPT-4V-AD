#!/usr/bin/env python3
"""
Auto-tune region sizing for grid/superpixel with top-k explained-difference constraint.

Pipeline per round:
1) region_division.py with candidate size lists (subset)
2) make_masks_overlay.py to compute region_diff_stats.csv + top-k summaries
3) select_region_sizing.py to summarize and recommend

Stops early when each target base method has at least one candidate with:
median_explain_ratio_topk >= min_ratio
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-tune region sizing from overlap statistics.")

    script_dir = Path(__file__).resolve().parent
    p.add_argument("--region-division-script", default=str(script_dir / "region_division.py"))
    p.add_argument("--overlay-script", default=str(script_dir / "make_masks_overlay.py"))
    p.add_argument("--selector-script", default=str(script_dir / "select_region_sizing.py"))

    p.add_argument("--input-dir", required=True, help="PNG spectrogram root for region_division.py")
    p.add_argument("--pairs-csv", required=True, help="real/fake pairs CSV for make_masks_overlay.py")
    p.add_argument("--spec-dir", required=True, help="spec image root for make_masks_overlay.py")
    p.add_argument("--work-dir", required=True, help="Root folder for tuning outputs.")

    p.add_argument("--subset-limit", type=int, default=500, help="Number of samples to evaluate per round.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--img-size", type=int, default=768)
    p.add_argument("--edge-pixel", type=int, default=1)

    p.add_argument("--grid-candidates", nargs="+", type=int, default=[10, 8, 6, 5, 4])
    p.add_argument("--superpixel-n-candidates", nargs="+", type=int, default=[100, 80, 60, 50, 40, 30])
    p.add_argument("--superpixel-compactness", nargs="+", type=float, default=[20.0])
    p.add_argument("--include-sam", action="store_true", default=False)

    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--min-ratio", type=float, default=0.5)
    p.add_argument("--target-methods", nargs="+", default=["grid", "superpixel"])
    p.add_argument("--max-rounds", type=int, default=None, help="Default=min(len(grid), len(superpixel_n)).")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def run_stream(cmd: List[str], prefix: str) -> None:
    print(f"[cmd:{prefix}] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{prefix}] {line.rstrip()}")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{prefix} failed with exit code {rc}")


def collect_generated_stems(region_out: Path) -> set[str]:
    stems: set[str] = set()
    for p in region_out.rglob("*_masks.pth"):
        method = p.parent.name
        suffix = f"_{method}_masks.pth"
        name = p.name
        if name.endswith(suffix):
            stems.add(name[: -len(suffix)])
    return stems


def build_subset_pairs_csv(src_pairs_csv: Path, stems: set[str], out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src_pairs_csv.open("r", encoding="utf-8", newline="") as fin, out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(["real_path", "fake_path"])
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#") or row[0].lower() == "real_path":
                continue
            if len(row) < 2:
                continue
            real_path = row[0].strip().strip('"').strip("'")
            fake_path = row[1].strip().strip('"').strip("'")
            if not fake_path:
                continue
            stem = Path(fake_path).stem
            if stem in stems:
                writer.writerow([real_path, fake_path])
                kept += 1
    return kept


def method_name_grid(grid_list: List[int]) -> List[str]:
    if len(grid_list) == 1:
        return ["grid"]
    return [f"grid_n{n}" for n in grid_list]


def method_name_superpixel(n_list: List[int], compactness: List[float]) -> List[str]:
    out = []
    for n in n_list:
        for c in compactness:
            out.append(f"superpixel_n{n}_c{c}")
    return out


def parse_method(method: str) -> Tuple[str, Optional[int], Optional[float]]:
    if method == "grid":
        return "grid", None, None
    if method.startswith("grid_n"):
        try:
            return "grid", int(method.split("grid_n", 1)[1]), None
        except Exception:
            return "grid", None, None
    if method.startswith("superpixel_n"):
        # superpixel_n{n}_c{compactness}
        try:
            body = method.split("superpixel_n", 1)[1]
            n_str, c_str = body.split("_c", 1)
            return "superpixel", int(n_str), float(c_str)
        except Exception:
            return "superpixel", None, None
    if method.startswith("sam"):
        return "sam", None, None
    return method, None, None


def choose_best_for_base(rows: List[dict], base: str, min_ratio: float) -> Optional[dict]:
    cands = []
    for r in rows:
        b, nseg, comp = parse_method(r["method"])
        if b != base:
            continue
        med = float(r["median_explain_ratio_topk"])
        pr = float(r["pass_rate_topk_ratio"])
        cands.append(
            {
                "method": r["method"],
                "median": med,
                "pass_rate": pr,
                "nseg": nseg,
                "compactness": comp,
            }
        )
    passing = [c for c in cands if c["median"] >= min_ratio]
    if not passing:
        return None
    if base in {"grid", "superpixel"}:
        passing = sorted(
            passing,
            key=lambda x: ((x["nseg"] if x["nseg"] is not None else -1), x["median"], x["pass_rate"]),
            reverse=True,
        )
    else:
        passing = sorted(passing, key=lambda x: (x["median"], x["pass_rate"]), reverse=True)
    return passing[0]


def load_method_summary(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    region_division_script = Path(args.region_division_script).expanduser().resolve()
    overlay_script = Path(args.overlay_script).expanduser().resolve()
    selector_script = Path(args.selector_script).expanduser().resolve()

    max_rounds_default = min(len(args.grid_candidates), len(args.superpixel_n_candidates))
    max_rounds = args.max_rounds if args.max_rounds is not None else max_rounds_default
    max_rounds = max(1, max_rounds)

    final_reco: Dict[str, dict] = {}
    round_summaries: List[dict] = []

    for r in range(1, max_rounds + 1):
        grid_list = args.grid_candidates[: min(r, len(args.grid_candidates))]
        sp_n_list = args.superpixel_n_candidates[: min(r, len(args.superpixel_n_candidates))]
        methods = method_name_grid(grid_list) + method_name_superpixel(sp_n_list, args.superpixel_compactness)
        region_methods = ["grid", "superpixel"]
        if args.include_sam:
            region_methods.append("sam")
            methods.append("sam")

        round_dir = work_dir / f"round_{r:02d}"
        region_out = round_dir / "regions"
        overlay_out = round_dir / "overlay"
        subset_pairs_csv = round_dir / "pairs_subset.csv"
        round_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[round {r}] grid={grid_list} superpixel_n={sp_n_list} compactness={args.superpixel_compactness}")

        cmd_region = [
            sys.executable,
            str(region_division_script),
            "--device", str(args.device),
            "--input-dir", str(args.input_dir),
            "--output-dir", str(region_out),
            "--region_division_methods", *region_methods,
            "--img_size", str(args.img_size),
            "--edge_pixel", str(args.edge_pixel),
            "--limit", str(args.subset_limit),
            "--grid-div-nums", *[str(x) for x in grid_list],
            "--slic-n-segments-list", *[str(x) for x in sp_n_list],
            "--slic-compactness", *[str(x) for x in args.superpixel_compactness],
        ]
        if args.overwrite:
            cmd_region.append("--overwrite")
        run_stream(cmd_region, prefix=f"region_division:r{r}")

        # Build a subset pairs file from stems that were actually generated in this round.
        stems = collect_generated_stems(region_out)
        kept_pairs = build_subset_pairs_csv(Path(args.pairs_csv), stems, subset_pairs_csv)
        print(f"[round {r}] subset_pairs={kept_pairs} stems={len(stems)} file={subset_pairs_csv}")
        if kept_pairs == 0:
            raise RuntimeError(
                f"round {r}: no pairs matched generated stems. "
                f"Check --input-dir and --pairs-csv stem compatibility."
            )

        cmd_overlay = [
            sys.executable,
            str(overlay_script),
            "--input_pairs", str(subset_pairs_csv),
            "--spec_dir", str(args.spec_dir),
            "--output_dir", str(overlay_out),
            "--region_outputs", str(region_out),
            "--region_methods", *methods,
            "--overlay_all_region_methods",
            "--topk-explain-k", str(args.topk),
            "--topk-explain-min-ratio", str(args.min_ratio),
        ]
        if args.overwrite:
            cmd_overlay.append("--overwrite")
        run_stream(cmd_overlay, prefix=f"make_masks_overlay:r{r}")

        diff_csv = overlay_out / "region_diff_stats.csv"
        cmd_selector = [
            sys.executable,
            str(selector_script),
            "--region-diff-csv", str(diff_csv),
            "--topk", str(args.topk),
            "--min-ratio", str(args.min_ratio),
            "--selection-mode", "max_granularity_pass",
            "--output-dir", str(overlay_out),
        ]
        run_stream(cmd_selector, prefix=f"select_region_sizing:r{r}")

        summary_csv = overlay_out / "topk_explain_ratio_per_method.csv"
        if not summary_csv.exists():
            raise RuntimeError(
                f"round {r}: missing method summary {summary_csv}. "
                "Overlay step may have produced no rows."
            )
        rows = load_method_summary(summary_csv)
        reco_this_round: Dict[str, dict] = {}
        for base in args.target_methods:
            reco = choose_best_for_base(rows, base=base, min_ratio=args.min_ratio)
            if reco is not None:
                reco_this_round[base] = reco
                print(
                    f"[round {r}] pass {base}: method={reco['method']} "
                    f"median={reco['median']:.4f} pass_rate={reco['pass_rate']:.4f}"
                )
            else:
                print(f"[round {r}] no passing candidate yet for {base}")

        round_summaries.append(
            {
                "round": r,
                "grid_candidates": grid_list,
                "superpixel_n_candidates": sp_n_list,
                "recommendations": reco_this_round,
                "summary_csv": str(summary_csv),
            }
        )

        for k, v in reco_this_round.items():
            final_reco[k] = v

        if all(m in final_reco for m in args.target_methods):
            print(f"[stop] all target methods satisfied by round {r}")
            break

    report = {
        "topk": args.topk,
        "min_ratio": args.min_ratio,
        "target_methods": args.target_methods,
        "final_recommendations": final_reco,
        "rounds": round_summaries,
    }
    out_json = work_dir / "auto_tune_region_sizing_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] report={out_json}")

    for base in args.target_methods:
        reco = final_reco.get(base)
        if reco is None:
            print(f"[final] {base}: no candidate reached median >= {args.min_ratio:.3f}")
            continue
        print(
            f"[final] {base}: method={reco['method']} "
            f"median={reco['median']:.4f} pass_rate={reco['pass_rate']:.4f}"
        )
        if base == "grid":
            if reco["nseg"] is not None:
                print(f"        region_division.py arg -> --div_num {reco['nseg']}")
        if base == "superpixel":
            if reco["nseg"] is not None and reco["compactness"] is not None:
                print(
                    f"        region_division.py args -> --slic-n-segments {reco['nseg']} "
                    f"--slic-compactness {reco['compactness']}"
                )


if __name__ == "__main__":
    main()
