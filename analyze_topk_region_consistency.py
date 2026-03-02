import argparse
import math
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


VOCODER_RE = re.compile(r"^(?P<vocoder>.+?)_LA_")
K = 3


def extract_vocoder(sample_id: str) -> str:
    m = VOCODER_RE.match(str(sample_id))
    if m:
        return m.group("vocoder")
    return str(sample_id).split("_", 1)[0]


def choose2(n: int) -> int:
    return 0 if n < 2 else (n * (n - 1)) // 2


def jaccard_from_overlap(m: int) -> float:
    return m / (2 * K - m)


def kuncheva_from_overlap(m: int, n_regions: int) -> float:
    if n_regions <= K:
        return 1.0 if m == K else 0.0
    expected = (K * K) / n_regions
    denom = K - expected
    if denom == 0:
        return 0.0
    return (m - expected) / denom


def p_value_greater(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    return float((np.sum(null_values >= observed) + 1) / (null_values.size + 1))


def z_vs_null(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    if null_values.size < 2:
        return math.nan
    sd = float(null_values.std(ddof=1))
    if sd == 0:
        return math.nan
    return float((observed - null_values.mean()) / sd)


def build_sample_sets(topk_df: pd.DataFrame) -> pd.DataFrame:
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
        .reset_index(name="region_set")
    )
    grouped["sample_key"] = grouped["sample_id"] + "||" + grouped["method"]
    grouped["vocoder"] = grouped["sample_id"].map(extract_vocoder)
    return grouped


def overlap_hist_within(set_counts: Counter) -> np.ndarray:
    hist = np.zeros(K + 1, dtype=np.int64)
    items = list(set_counts.items())
    sets = [(set(key), count) for key, count in items]

    for i, (set_i, count_i) in enumerate(sets):
        hist[K] += choose2(count_i)
        for set_j, count_j in sets[i + 1:]:
            m = len(set_i & set_j)
            hist[m] += count_i * count_j
    return hist


def overlap_hist_between(set_counts_a: Counter, set_counts_b: Counter) -> np.ndarray:
    hist = np.zeros(K + 1, dtype=np.int64)
    items_a = [(set(key), count) for key, count in set_counts_a.items()]
    items_b = [(set(key), count) for key, count in set_counts_b.items()]

    for set_a, count_a in items_a:
        for set_b, count_b in items_b:
            m = len(set_a & set_b)
            hist[m] += count_a * count_b
    return hist


def hist_total(hist: np.ndarray) -> int:
    return int(hist.sum())


def hist_mean(hist: np.ndarray, metric_values: np.ndarray) -> float:
    total = hist_total(hist)
    if total == 0:
        return math.nan
    return float(np.dot(hist, metric_values) / total)


def hist_median(hist: np.ndarray, metric_values: np.ndarray) -> float:
    total = hist_total(hist)
    if total == 0:
        return math.nan
    order = np.argsort(metric_values)
    sorted_hist = hist[order]
    sorted_vals = metric_values[order]
    csum = np.cumsum(sorted_hist)
    idx = int(np.searchsorted(csum, (total + 1) / 2, side="left"))
    return float(sorted_vals[idx])


def hist_probs(hist: np.ndarray) -> np.ndarray:
    total = hist_total(hist)
    if total == 0:
        return np.full(hist.shape, np.nan)
    return hist / total


def bootstrap_mean_ci_from_hist(
    hist: np.ndarray,
    metric_values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 1337,
    alpha: float = 0.05,
) -> tuple[float, float]:
    total = hist_total(hist)
    if total == 0:
        return (math.nan, math.nan)
    probs = hist_probs(hist)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.multinomial(total, probs)
        means[i] = float(np.dot(draw, metric_values) / total)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def random_overlap_probs(n_regions: int) -> np.ndarray:
    denom = math.comb(n_regions, K)
    probs = np.zeros(K + 1, dtype=float)
    for m in range(K + 1):
        if K - m > n_regions - K:
            continue
        probs[m] = (math.comb(K, m) * math.comb(n_regions - K, K - m)) / denom
    return probs


def null_mean_distribution(
    n_pairs: int,
    overlap_probs: np.ndarray,
    metric_values: np.ndarray,
    n_perm: int = 5000,
    seed: int = 1337,
) -> np.ndarray:
    if n_pairs <= 0:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        draw = rng.multinomial(n_pairs, overlap_probs)
        out[i] = float(np.dot(draw, metric_values) / n_pairs)
    return out


def gap_null_distribution(
    within_pairs: int,
    between_pairs: int,
    pooled_probs: np.ndarray,
    metric_values: np.ndarray,
    n_perm: int = 5000,
    seed: int = 1337,
) -> np.ndarray:
    if within_pairs <= 0 or between_pairs <= 0:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        within_draw = rng.multinomial(within_pairs, pooled_probs)
        between_draw = rng.multinomial(between_pairs, pooled_probs)
        within_mean = float(np.dot(within_draw, metric_values) / within_pairs)
        between_mean = float(np.dot(between_draw, metric_values) / between_pairs)
        out[i] = within_mean - between_mean
    return out


def sum_hists(hists: list[np.ndarray]) -> np.ndarray:
    if not hists:
        return np.zeros(K + 1, dtype=np.int64)
    out = np.zeros(K + 1, dtype=np.int64)
    for hist in hists:
        out += hist
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk-csv", required=True)
    ap.add_argument("--stats-csv", required=True)
    ap.add_argument("--method", default=None, help="Optional method filter, e.g. grid")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    topk_df = pd.read_csv(args.topk_csv)
    stats_df = pd.read_csv(args.stats_csv)

    required_topk = {"sample_id", "method", "region_id"}
    required_stats = {"image", "method", "region_id"}
    if not required_topk.issubset(topk_df.columns):
        missing = sorted(required_topk - set(topk_df.columns))
        raise ValueError(f"Missing topk columns: {missing}")
    if not required_stats.issubset(stats_df.columns):
        missing = sorted(required_stats - set(stats_df.columns))
        raise ValueError(f"Missing stats columns: {missing}")

    if args.method is not None:
        topk_df = topk_df[topk_df["method"] == args.method].copy()
        stats_df = stats_df[stats_df["method"] == args.method].copy()

    sample_df = build_sample_sets(topk_df)
    n_regions = int(stats_df["region_id"].nunique())
    if n_regions < K:
        raise ValueError(f"Expected at least {K} unique regions, found {n_regions}")

    print(f"n_samples={len(sample_df)}")
    print(f"n_vocoders={sample_df['vocoder'].nunique()}")
    print(f"n_regions={n_regions}")
    print("")

    j_vals = np.array([jaccard_from_overlap(m) for m in range(K + 1)], dtype=float)
    k_vals = np.array([kuncheva_from_overlap(m, n_regions) for m in range(K + 1)], dtype=float)
    random_probs = random_overlap_probs(n_regions)

    set_counts_by_vocoder: dict[str, Counter] = {}
    sample_counts_by_vocoder: dict[str, int] = {}
    for vocoder, voc_df in sample_df.groupby("vocoder", sort=True):
        counter = Counter(voc_df["region_set"].tolist())
        set_counts_by_vocoder[vocoder] = counter
        sample_counts_by_vocoder[vocoder] = int(voc_df.shape[0])

    pooled_counts = Counter(sample_df["region_set"].tolist())
    pooled_pair_hist = overlap_hist_within(pooled_counts)
    pooled_pair_probs = hist_probs(pooled_pair_hist)

    vocoders = sorted(set_counts_by_vocoder)
    between_hist_by_vocoder: dict[str, np.ndarray] = {}
    for target in vocoders:
        target_counts = set_counts_by_vocoder[target]
        target_hist_parts = []
        for other in vocoders:
            if other == target:
                continue
            target_hist_parts.append(overlap_hist_between(target_counts, set_counts_by_vocoder[other]))
        between_hist_by_vocoder[target] = sum_hists(target_hist_parts)

    for vocoder in vocoders:
        n_samples = sample_counts_by_vocoder[vocoder]
        if n_samples < 2:
            print(f"vocoder={vocoder}, n_samples={n_samples}, skipped=need_at_least_2_samples")
            continue

        within_hist = overlap_hist_within(set_counts_by_vocoder[vocoder])
        within_pairs = hist_total(within_hist)
        between_hist = between_hist_by_vocoder[vocoder]
        between_pairs = hist_total(between_hist)

        j_mean = hist_mean(within_hist, j_vals)
        j_median = hist_median(within_hist, j_vals)
        k_mean = hist_mean(within_hist, k_vals)
        k_median = hist_median(within_hist, k_vals)
        k_ci_lo, k_ci_hi = bootstrap_mean_ci_from_hist(
            within_hist,
            k_vals,
            n_boot=args.n_boot,
            seed=args.seed,
        )

        j_null = null_mean_distribution(
            within_pairs,
            random_probs,
            j_vals,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        k_null = null_mean_distribution(
            within_pairs,
            random_probs,
            k_vals,
            n_perm=args.n_perm,
            seed=args.seed,
        )

        j_between_mean = hist_mean(between_hist, j_vals)
        k_between_mean = hist_mean(between_hist, k_vals)
        j_gap = j_mean - j_between_mean if not math.isnan(j_between_mean) else math.nan
        k_gap = k_mean - k_between_mean if not math.isnan(k_between_mean) else math.nan

        j_gap_null = gap_null_distribution(
            within_pairs,
            between_pairs,
            pooled_pair_probs,
            j_vals,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        k_gap_null = gap_null_distribution(
            within_pairs,
            between_pairs,
            pooled_pair_probs,
            k_vals,
            n_perm=args.n_perm,
            seed=args.seed,
        )

        print(
            f"vocoder={vocoder}, "
            f"n_samples={n_samples}, "
            f"n_pairs={within_pairs}, "
            f"j_mean={j_mean:.6f}, "
            f"j_median={j_median:.6f}, "
            f"j_perm_p={p_value_greater(j_mean, j_null):.6f}, "
            f"j_effect_z={z_vs_null(j_mean, j_null):.6f}, "
            f"j_within_minus_between={j_gap:.6f}, "
            f"j_gap_perm_p={p_value_greater(j_gap, j_gap_null):.6f}, "
            f"k_mean={k_mean:.6f}, "
            f"k_median={k_median:.6f}, "
            f"k_boot_ci_lo={k_ci_lo:.6f}, "
            f"k_boot_ci_hi={k_ci_hi:.6f}, "
            f"k_perm_p={p_value_greater(k_mean, k_null):.6f}, "
            f"k_effect_z={z_vs_null(k_mean, k_null):.6f}, "
            f"k_within_minus_between={k_gap:.6f}, "
            f"k_gap_perm_p={p_value_greater(k_gap, k_gap_null):.6f}"
        )


if __name__ == "__main__":
    main()
