import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


VOCODER_RE = re.compile(r"^(?P<vocoder>.+?)_LA_")
K = 3
JACCARD_BY_OVERLAP = {0: 0.0, 1: 0.2, 2: 0.5, 3: 1.0}
VOCODER_DISPLAY_NAMES = {
    "hifigan": "HiFiGAN",
    "hn-sinc-nsf": "Hn-NSF",
    "hn-sinc-nsf-hifi": "NSF-HiFiGAN",
    "waveglow": "WaveGlow",
}
OVERLAP_COLORS = {
    0: "#DCEAF6",  # very light blue
    1: "#AFCFE8",  # light blue
    2: "#6FADD8",  # medium blue
    3: "#2F6FA3",  # dark blue
}
OVERLAP_HATCHES = {
    0: None,
    1: "///",
    2: None,
    3: "xx",
}


def extract_vocoder(sample_id: str) -> str:
    m = VOCODER_RE.match(str(sample_id))
    if m:
        return m.group("vocoder")
    return str(sample_id).split("_", 1)[0]


def choose2(n: int) -> int:
    return 0 if n < 2 else (n * (n - 1)) // 2


def build_sample_triples(topk_df: pd.DataFrame) -> pd.DataFrame:
    counts = topk_df.groupby(["sample_id", "method"])["region_id"].nunique()
    bad = counts[counts != K]
    if not bad.empty:
        raise ValueError(
            f"Each (sample_id, method) must have exactly {K} unique region_id values. "
            f"Found mismatches for {len(bad)} groups."
        )

    grouped = (
        topk_df.groupby(["sample_id", "method"])["region_id"]
        .apply(lambda s: tuple(sorted(int(x) for x in s.tolist())))
        .reset_index(name="triple")
    )
    grouped["vocoder"] = grouped["sample_id"].map(extract_vocoder)
    return grouped


def exact_overlap_counts_from_triples(triples: list[tuple[int, int, int]]) -> dict:
    n = len(triples)
    triple_counts = Counter(triples)
    pair_counts = Counter()
    region_counts = Counter()

    for triple, count in triple_counts.items():
        a, b, c = triple
        pair_counts[(a, b)] += count
        pair_counts[(a, c)] += count
        pair_counts[(b, c)] += count
        region_counts[a] += count
        region_counts[b] += count
        region_counts[c] += count

    c3 = sum(choose2(count) for count in triple_counts.values())
    c2 = sum(choose2(count) for count in pair_counts.values()) - (3 * c3)
    c1 = sum(choose2(count) for count in region_counts.values()) - (2 * c2) - (3 * c3)
    total_pairs = choose2(n)
    c0 = total_pairs - (c1 + c2 + c3)

    counts = {0: c0, 1: c1, 2: c2, 3: c3}
    if min(counts.values()) < 0:
        raise ValueError(
            "Computed a negative overlap count. Check that each sample contributes exactly 3 unique region IDs."
        )

    fracs = {r: (counts[r] / total_pairs if total_pairs else math.nan) for r in range(4)}
    return {
        "n_samples": n,
        "n_pairs": total_pairs,
        "counts": counts,
        "fractions": fracs,
    }


def write_outputs(results: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    calc_rows = []

    for vocoder, stats in results.items():
        summary_rows.append(
            {
                "vocoder": vocoder,
                "n_samples": stats["n_samples"],
                "n_pairs": stats["n_pairs"],
                "frac_overlap_0": stats["fractions"][0],
                "frac_overlap_1": stats["fractions"][1],
                "frac_overlap_2": stats["fractions"][2],
                "frac_overlap_3": stats["fractions"][3],
            }
        )

        for overlap in range(4):
            calc_rows.append(
                {
                    "vocoder": vocoder,
                    "n_samples": stats["n_samples"],
                    "n_pairs": stats["n_pairs"],
                    "overlap_level": overlap,
                    "jaccard": JACCARD_BY_OVERLAP[overlap],
                    "pair_count": stats["counts"][overlap],
                    "pair_fraction": stats["fractions"][overlap],
                }
            )

    summary_path = out_dir / "top3_overlap_stacked_bar_summary.csv"
    calc_path = out_dir / "top3_overlap_stacked_bar_long.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(calc_rows).to_csv(calc_path, index=False)
    return summary_path, calc_path


def print_reproducible_calculation(results: dict) -> None:
    print("# Exact overlap-level fractions for stacked bars")
    print("# overlap 0 -> J=0.0, overlap 1 -> J=0.2, overlap 2 -> J=0.5, overlap 3 -> J=1.0")
    print("")
    for vocoder, stats in results.items():
        print(f"vocoder={vocoder}")
        print(f"  n_samples={stats['n_samples']}")
        print(f"  n_pairs=C(n,2)={stats['n_pairs']}")
        for overlap in range(4):
            print(
                f"  overlap_{overlap}: "
                f"count={stats['counts'][overlap]}, "
                f"fraction={stats['fractions'][overlap]:.12f}, "
                f"jaccard={JACCARD_BY_OVERLAP[overlap]:.1f}"
            )
        print("")


def maybe_plot(results: dict, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    vocoders = list(results.keys())
    display_labels = [VOCODER_DISPLAY_NAMES.get(v, v) for v in vocoders]
    frac0 = [results[v]["fractions"][0] * 100.0 for v in vocoders]
    frac1 = [results[v]["fractions"][1] * 100.0 for v in vocoders]
    frac2 = [results[v]["fractions"][2] * 100.0 for v in vocoders]
    frac3 = [results[v]["fractions"][3] * 100.0 for v in vocoders]
    segment_values = [frac0, frac1, frac2, frac3]

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    ax.bar(
        display_labels,
        frac0,
        label="Overlap 0 (J=0.0)",
        color=OVERLAP_COLORS[0],
        edgecolor="#4A4A4A",
        linewidth=0.4,
        hatch=OVERLAP_HATCHES[0],
        width=0.62,
    )
    ax.bar(
        display_labels,
        frac1,
        bottom=frac0,
        label="Overlap 1 (J=0.2)",
        color=OVERLAP_COLORS[1],
        edgecolor="#4A4A4A",
        linewidth=0.4,
        hatch=OVERLAP_HATCHES[1],
        width=0.62,
    )
    bottom2 = [a + b for a, b in zip(frac0, frac1)]
    ax.bar(
        display_labels,
        frac2,
        bottom=bottom2,
        label="Overlap 2 (J=0.5)",
        color=OVERLAP_COLORS[2],
        edgecolor="#4A4A4A",
        linewidth=0.4,
        hatch=OVERLAP_HATCHES[2],
        width=0.62,
    )
    bottom3 = [a + b + c for a, b, c in zip(frac0, frac1, frac2)]
    ax.bar(
        display_labels,
        frac3,
        bottom=bottom3,
        label="Overlap 3 (J=1.0)",
        color=OVERLAP_COLORS[3],
        edgecolor="#4A4A4A",
        linewidth=0.4,
        hatch=OVERLAP_HATCHES[3],
        width=0.62,
    )

    ax.set_ylabel("Within-vocoder pairs (%)", fontsize=17)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        columnspacing=1.4,
        handlelength=1.3,
        fontsize=14,
    )
    plt.xticks(rotation=0, ha="center")

    for idx, vocoder in enumerate(vocoders):
        values = [segment_values[level][idx] for level in range(4)]
        dominant_level = max(range(4), key=lambda level: values[level])
        lower = sum(values[:dominant_level])
        height = values[dominant_level]
        y = lower + (height / 2.0)
        label = f"{height:.1f}%"
        ax.text(
            idx,
            y,
            label,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="semibold",
            color="black",
        )

    fig.subplots_adjust(top=0.84)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute exact overlap-level fractions for stacked bars of within-vocoder top-3 region overlap."
    )
    ap.add_argument("--topk-csv", required=True, help="CSV with sample_id, method, region_id for top-3 regions.")
    ap.add_argument("--method", default=None, help="Optional method filter, e.g. grid")
    ap.add_argument(
        "--out-dir",
        default=".",
        help="Directory to save CSV outputs and optional plot.",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Also save a stacked bar PNG (requires matplotlib).",
    )
    args = ap.parse_args()

    topk_df = pd.read_csv(args.topk_csv)
    required = {"sample_id", "method", "region_id"}
    if not required.issubset(topk_df.columns):
        missing = sorted(required - set(topk_df.columns))
        raise ValueError(f"Missing topk columns: {missing}")

    if args.method is not None:
        topk_df = topk_df[topk_df["method"] == args.method].copy()

    sample_df = build_sample_triples(topk_df)
    results = {}
    for vocoder, voc_df in sample_df.groupby("vocoder", sort=True):
        triples = voc_df["triple"].tolist()
        if len(triples) < 2:
            continue
        results[vocoder] = exact_overlap_counts_from_triples(triples)

    if not results:
        raise ValueError("No vocoder had at least 2 samples after filtering.")

    print_reproducible_calculation(results)

    out_dir = Path(args.out_dir)
    summary_path, calc_path = write_outputs(results, out_dir)
    print(f"saved_summary_csv={summary_path}")
    print(f"saved_long_csv={calc_path}")

    if args.plot:
        plot_path = out_dir / "top3_overlap_stacked_bar.png"
        saved_plot = maybe_plot(results, plot_path)
        print(f"saved_plot={saved_plot}")


if __name__ == "__main__":
    main()
