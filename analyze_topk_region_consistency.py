import argparse
import math
import re
from itertools import combinations

import numpy as np
import pandas as pd


VOCODER_RE = re.compile(r"^(?P<vocoder>.+?)_LA_")


def extract_vocoder(sample_id: str) -> str:
    m = VOCODER_RE.match(str(sample_id))
    if m:
        return m.group("vocoder")
    return str(sample_id).split("_", 1)[0]


def jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def kuncheva(a: set, b: set, k: int, n_regions: int) -> float:
    if n_regions <= k:
        return 1.0 if a == b else 0.0
    expected = (k * k) / n_regions
    denom = k - expected
    if denom == 0:
        return 0.0
    return (len(a & b) - expected) / denom


def bootstrap_ci(values, n_boot=2000, seed=1337, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = []
    n = values.size
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(float(sample.mean()))
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def permutation_null_mean(k, n_regions, n_pairs, metric, n_perm=5000, seed=1337):
    rng = np.random.default_rng(seed)
    region_ids = np.arange(n_regions)
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        vals = []
        for _ in range(n_pairs):
            a = set(rng.choice(region_ids, size=k, replace=False).tolist())
            b = set(rng.choice(region_ids, size=k, replace=False).tolist())
            vals.append(metric(a, b))
        out[i] = np.mean(vals)
    return out


def gap_permutation(all_samples_df, target_vocoder, sample_sets, metric_fn, n_perm=5000, seed=1337):
    rng = np.random.default_rng(seed)
    labels = all_samples_df["vocoder"].to_numpy()
    sample_ids = all_samples_df["sample_key"].to_numpy()
    target_n = int((labels == target_vocoder).sum())
    out = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        within_ids = sample_ids[shuffled == target_vocoder][:target_n]
        within_set = set(within_ids.tolist())
        between_ids = sample_ids[[sid not in within_set for sid in sample_ids]]

        within_vals = [
            metric_fn(sample_sets[a], sample_sets[b])
            for a, b in combinations(within_ids, 2)
        ]
        between_vals = [
            metric_fn(sample_sets[a], sample_sets[b])
            for a in within_ids
            for b in between_ids
        ]
        out[i] = np.mean(within_vals) - np.mean(between_vals)
    return out


def p_value_greater(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return float((np.sum(null_values >= observed) + 1) / (null_values.size + 1))


def z_vs_null(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)
    if null_values.size < 2:
        return math.nan
    sd = float(null_values.std(ddof=1))
    if sd == 0:
        return math.nan
    return float((observed - null_values.mean()) / sd)


def build_sample_sets(topk_df: pd.DataFrame) -> pd.DataFrame:
    counts = topk_df.groupby(["sample_id", "method"])["region_id"].nunique()
    bad = counts[counts != 3]
    if not bad.empty:
        raise ValueError(
            "Each (sample_id, method) must have exactly 3 unique region_id values. "
            f"Found mismatches for {len(bad)} groups."
        )

    grouped = (
        topk_df.groupby(["sample_id", "method"])["region_id"]
        .apply(lambda s: frozenset(int(x) for x in s.tolist()))
        .reset_index(name="region_set")
    )
    grouped["sample_key"] = grouped["sample_id"] + "||" + grouped["method"]
    grouped["vocoder"] = grouped["sample_id"].map(extract_vocoder)
    return grouped


def main():
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
    if n_regions < 3:
        raise ValueError(f"Expected at least 3 unique regions, found {n_regions}")

    sample_sets = dict(zip(sample_df["sample_key"], sample_df["region_set"]))

    print(f"n_samples={len(sample_df)}")
    print(f"n_vocoders={sample_df['vocoder'].nunique()}")
    print(f"n_regions={n_regions}")
    print("")

    all_between_j = []
    all_between_k = []
    rows = sample_df[["sample_key", "vocoder"]].to_records(index=False)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i].vocoder == rows[j].vocoder:
                continue
            a = sample_sets[rows[i].sample_key]
            b = sample_sets[rows[j].sample_key]
            all_between_j.append(jaccard(a, b))
            all_between_k.append(kuncheva(a, b, 3, n_regions))

    for vocoder, voc_df in sample_df.groupby("vocoder", sort=True):
        keys = voc_df["sample_key"].tolist()
        if len(keys) < 2:
            print(f"vocoder={vocoder}, n_samples={len(keys)}, skipped=need_at_least_2_samples")
            continue

        within_j = []
        within_k = []
        for a_key, b_key in combinations(keys, 2):
            a = sample_sets[a_key]
            b = sample_sets[b_key]
            within_j.append(jaccard(a, b))
            within_k.append(kuncheva(a, b, 3, n_regions))

        j_mean = float(np.mean(within_j))
        j_median = float(np.median(within_j))
        k_mean = float(np.mean(within_k))
        k_median = float(np.median(within_k))
        k_ci_lo, k_ci_hi = bootstrap_ci(within_k, n_boot=args.n_boot, seed=args.seed)

        j_null = permutation_null_mean(3, n_regions, len(within_j), jaccard, n_perm=args.n_perm, seed=args.seed)
        k_null = permutation_null_mean(
            3,
            n_regions,
            len(within_k),
            lambda a, b: kuncheva(a, b, 3, n_regions),
            n_perm=args.n_perm,
            seed=args.seed,
        )

        j_gap = j_mean - float(np.mean(all_between_j)) if all_between_j else math.nan
        k_gap = k_mean - float(np.mean(all_between_k)) if all_between_k else math.nan

        j_gap_null = gap_permutation(
            sample_df[["sample_key", "vocoder"]],
            vocoder,
            sample_sets,
            jaccard,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        k_gap_null = gap_permutation(
            sample_df[["sample_key", "vocoder"]],
            vocoder,
            sample_sets,
            lambda a, b: kuncheva(a, b, 3, n_regions),
            n_perm=args.n_perm,
            seed=args.seed,
        )

        print(
            f"vocoder={vocoder}, "
            f"n_samples={len(keys)}, "
            f"n_pairs={len(within_j)}, "
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
