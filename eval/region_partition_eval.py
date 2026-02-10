import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


EPS = 1e-9


@dataclass
class ScoreWeights:
    coverage: float
    low_overlap: float
    compactness: float
    geom_distinctiveness: float
    size_balance: float
    region_count: float
    mask_info: float
    freq_distinctiveness: float
    mask_distinctiveness: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate region_division.py outputs (*.pth masks) and compare methods."
    )
    parser.add_argument(
        "--masks-root",
        type=str,
        required=True,
        help="Root directory containing method subfolders with *_masks.pth files.",
    )
    parser.add_argument(
        "--method-glob",
        type=str,
        default="*",
        help="Glob for method folder names under --masks-root.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output prefix path. Defaults to <masks-root>/region_partition_eval.",
    )

    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Optional root containing source spectrogram images for frequency-energy metrics.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".png",
        help="Image extension under --image-root (default: .png).",
    )

    parser.add_argument(
        "--overlay-root",
        type=str,
        default=None,
        help="Optional root with make_masks_overlay outputs (<stem>_gt_binary.npy / _gt_continuous.npy).",
    )
    parser.add_argument(
        "--overlay-kind",
        choices=["auto", "binary", "continuous"],
        default="auto",
        help="Which overlay map type to use when both exist.",
    )
    parser.add_argument(
        "--overlay-suffix-binary",
        type=str,
        default="_gt_binary.npy",
        help="Suffix for binary overlay files.",
    )
    parser.add_argument(
        "--overlay-suffix-continuous",
        type=str,
        default="_gt_continuous.npy",
        help="Suffix for continuous overlay files.",
    )

    parser.add_argument(
        "--target-regions",
        type=float,
        default=50.0,
        help="Target number of regions used in region-count score.",
    )
    parser.add_argument(
        "--topk-fraction",
        type=float,
        default=0.1,
        help="Top-k fraction for mask informativeness (k=max(1, round(frac*K))).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=2000,
        help="Max random region pairs per sample for pairwise IoU/distinctiveness.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for pair subsampling.",
    )
    parser.add_argument(
        "--vocoder-sep-token",
        type=str,
        default="_LA_",
        help="Filename token used to infer vocoder from base stem.",
    )

    parser.add_argument(
        "--min-mask-info",
        type=float,
        default=0.0,
        help="Constraint threshold for mask_info_score in recommended configuration.",
    )
    parser.add_argument(
        "--min-total-distinctiveness",
        type=float,
        default=0.0,
        help="Constraint threshold for distinctiveness_total in recommended configuration.",
    )

    parser.add_argument("--w-coverage", type=float, default=0.15)
    parser.add_argument("--w-low-overlap", type=float, default=0.10)
    parser.add_argument("--w-compactness", type=float, default=0.10)
    parser.add_argument("--w-geom-distinctiveness", type=float, default=0.10)
    parser.add_argument("--w-size-balance", type=float, default=0.10)
    parser.add_argument("--w-region-count", type=float, default=0.15)
    parser.add_argument("--w-mask-info", type=float, default=0.20)
    parser.add_argument("--w-freq-distinctiveness", type=float, default=0.05)
    parser.add_argument("--w-mask-distinctiveness", type=float, default=0.05)

    parser.add_argument(
        "--phoneme-align-csv",
        type=str,
        default=None,
        help="Optional placeholder for future phoneme-alignment metrics (not used yet).",
    )

    return parser.parse_args()


def infer_base_name(mask_path: str, method: str) -> str:
    stem = os.path.basename(mask_path).replace("_masks.pth", "")
    suffix = f"_{method}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def infer_vocoder(base_name: str, sep_token: str) -> str:
    if sep_token and sep_token in base_name:
        return base_name.split(sep_token, 1)[0]
    return base_name.split("_", 1)[0]


def to_bool_array(mask) -> np.ndarray:
    if torch is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={mask.shape}")
    return mask.astype(bool)


def safe_norm(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    if s <= 0:
        return np.zeros_like(x, dtype=float)
    return x / (s + EPS)


def normalize_01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros_like(a, dtype=float)
    out = np.zeros_like(a, dtype=float)
    vals = a[finite]
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    if mx - mn < EPS:
        out[finite] = 0.0
        return out
    out[finite] = (vals - mn) / (mx - mn)
    return out


def binary_perimeter(mask: np.ndarray) -> float:
    mask_u8 = mask.astype(np.uint8)
    horizontal = np.sum(mask_u8[:, 1:] != mask_u8[:, :-1])
    vertical = np.sum(mask_u8[1:, :] != mask_u8[:-1, :])
    border = (
        np.sum(mask_u8[0, :])
        + np.sum(mask_u8[-1, :])
        + np.sum(mask_u8[:, 0])
        + np.sum(mask_u8[:, -1])
    )
    return float(horizontal + vertical + border)


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or np.allclose(x, 0.0):
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * np.sum(x) + EPS))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = safe_norm(np.asarray(p, dtype=float))
    q = safe_norm(np.asarray(q, dtype=float))
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + EPS) / (b[mask] + EPS))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def sample_pairs(n: int, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        return np.empty((0, 2), dtype=int)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return np.asarray(pairs, dtype=int)

    pairs = set()
    while len(pairs) < max_pairs:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((int(i), int(j)))
    return np.asarray(list(pairs), dtype=int)


def pairwise_mean(values: List[float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == out_h and w == out_w:
        return arr
    y_idx = np.clip((np.arange(out_h) * h / out_h).astype(int), 0, h - 1)
    x_idx = np.clip((np.arange(out_w) * w / out_w).astype(int), 0, w - 1)
    return arr[np.ix_(y_idx, x_idx)]


def load_image_gray(image_path: str) -> Optional[np.ndarray]:
    if Image is None:
        return None
    try:
        img = Image.open(image_path).convert("L")
        return np.asarray(img, dtype=float) / 255.0
    except Exception:
        return None


def load_masks(mask_path: str) -> List[np.ndarray]:
    if torch is None:
        raise SystemExit(
            "Missing dependency: torch. Install it (e.g., `pip install torch`) to run this script."
        )
    blob = torch.load(mask_path, map_location="cpu")
    if not isinstance(blob, dict) or "masks" not in blob:
        raise ValueError(f"Unexpected file format in {mask_path}")
    return [to_bool_array(m) for m in blob["masks"]]


def build_index(root: Optional[str], pattern: str) -> Dict[str, str]:
    if not root:
        return {}
    files = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    out: Dict[str, str] = {}
    for path in sorted(files):
        stem = os.path.splitext(os.path.basename(path))[0]
        out.setdefault(stem, path)
    return out


def collect_mask_files(masks_root: str, method_glob: str) -> List[Tuple[str, str]]:
    pattern = os.path.join(masks_root, method_glob, "*_masks.pth")
    files = sorted(glob.glob(pattern))
    return [(os.path.basename(os.path.dirname(p)), p) for p in files]


def geometric_metrics(
    masks: List[np.ndarray],
    target_regions: float,
    max_pairs: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    if len(masks) == 0:
        return {
            "n_regions": 0.0,
            "coverage": 0.0,
            "overlap_ratio": 1.0,
            "mean_compactness": 0.0,
            "mean_pairwise_iou": 1.0,
            "geom_distinctiveness": 0.0,
            "size_balance": 0.0,
            "region_count_score": 0.0,
            "mean_area": 0.0,
            "std_area": 0.0,
            "cv_area": 0.0,
        }

    h, w = masks[0].shape
    total_pixels = float(h * w)
    areas = np.array([float(m.sum()) for m in masks], dtype=float)

    sum_area = float(areas.sum())
    union_mask = np.zeros((h, w), dtype=bool)
    for m in masks:
        union_mask |= m
    union_area = float(union_mask.sum())
    coverage = union_area / (total_pixels + EPS)
    overlap_ratio = max(0.0, (sum_area - union_area) / (sum_area + EPS))

    perimeters = np.array([binary_perimeter(m) for m in masks], dtype=float)
    mean_compactness = float(np.mean((4.0 * math.pi * areas) / (perimeters * perimeters + EPS)))

    centroids = []
    for m in masks:
        ys, xs = np.where(m)
        if len(xs) == 0:
            centroids.append((0.0, 0.0))
        else:
            centroids.append((float(np.mean(xs)), float(np.mean(ys))))
    centroids = np.asarray(centroids, dtype=float)

    pairs = sample_pairs(len(masks), max_pairs=max_pairs, rng=rng)
    ious = []
    dists = []
    diag = math.sqrt(h * h + w * w) + EPS
    for i, j in pairs:
        inter = float(np.logical_and(masks[i], masks[j]).sum())
        uni = float(np.logical_or(masks[i], masks[j]).sum())
        ious.append(inter / (uni + EPS))

        dx = centroids[i, 0] - centroids[j, 0]
        dy = centroids[i, 1] - centroids[j, 1]
        dists.append(math.sqrt(dx * dx + dy * dy) / diag)

    region_count_score = float(
        math.exp(-abs(len(masks) - target_regions) / max(target_regions, 1.0))
    )

    mean_area = float(np.mean(areas))
    std_area = float(np.std(areas))
    cv_area = float(std_area / (mean_area + EPS))
    size_balance = float(1.0 - max(0.0, min(1.0, gini(areas))))

    return {
        "n_regions": float(len(masks)),
        "coverage": coverage,
        "overlap_ratio": overlap_ratio,
        "mean_compactness": mean_compactness,
        "mean_pairwise_iou": pairwise_mean(ious),
        "geom_distinctiveness": pairwise_mean(dists),
        "size_balance": size_balance,
        "region_count_score": region_count_score,
        "mean_area": mean_area,
        "std_area": std_area,
        "cv_area": cv_area,
    }


def frequency_metrics(masks: List[np.ndarray], image_gray: Optional[np.ndarray], max_pairs: int, rng: np.random.Generator) -> Dict[str, float]:
    if len(masks) < 2:
        return {
            "freq_geo_jsd": 0.0,
            "freq_energy_jsd": np.nan if image_gray is None else 0.0,
        }

    h, w = masks[0].shape
    pairs = sample_pairs(len(masks), max_pairs=max_pairs, rng=rng)

    freq_profiles_geo = []
    for m in masks:
        prof = np.sum(m.astype(float), axis=1)
        freq_profiles_geo.append(safe_norm(prof))

    geo_jsd = []
    for i, j in pairs:
        geo_jsd.append(js_divergence(freq_profiles_geo[i], freq_profiles_geo[j]))

    if image_gray is None:
        energy_jsd = np.nan
    else:
        img = image_gray
        if img.shape != (h, w):
            img = resize_nearest(img, h, w)
        freq_profiles_energy = []
        for m in masks:
            prof = np.sum((img * m.astype(float)), axis=1)
            freq_profiles_energy.append(safe_norm(prof))
        vals = []
        for i, j in pairs:
            vals.append(js_divergence(freq_profiles_energy[i], freq_profiles_energy[j]))
        energy_jsd = pairwise_mean(vals)

    return {
        "freq_geo_jsd": pairwise_mean(geo_jsd),
        "freq_energy_jsd": energy_jsd,
    }


def overlay_metrics(
    masks: List[np.ndarray],
    overlay_map: Optional[np.ndarray],
    topk_fraction: float,
    max_pairs: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    if overlay_map is None:
        return {
            "overlay_available": 0.0,
            "mask_info_topk_mass": np.nan,
            "mask_info_topk_iou": np.nan,
            "mask_info_score": np.nan,
            "mask_distinctiveness": np.nan,
        }

    h, w = masks[0].shape
    ov = np.asarray(overlay_map)
    if ov.ndim != 2:
        raise ValueError(f"Overlay map must be 2D, got {ov.shape}")
    if ov.shape != (h, w):
        ov = resize_nearest(ov, h, w)

    ov = normalize_01(ov)
    ov_bin = ov > 0.5
    total_mass = float(np.sum(ov))

    areas = np.array([float(m.sum()) for m in masks], dtype=float)
    masses = np.array([float(np.sum(ov * m.astype(float))) for m in masks], dtype=float)
    densities = masses / (areas + EPS)

    inters = np.array([float(np.logical_and(m, ov_bin).sum()) for m in masks], dtype=float)
    unions = np.array([float(np.logical_or(m, ov_bin).sum()) for m in masks], dtype=float)
    ious = inters / (unions + EPS)

    k = max(1, int(round(topk_fraction * len(masks))))
    order = np.argsort(-masses)
    topk = order[:k]

    topk_mass = float(np.sum(masses[topk])) / (total_mass + EPS) if total_mass > 0 else 0.0
    topk_iou = float(np.mean(ious[topk])) if len(topk) > 0 else 0.0
    mask_info_score = 0.5 * topk_mass + 0.5 * topk_iou

    pairs = sample_pairs(len(masks), max_pairs=max_pairs, rng=rng)
    density_diffs = [abs(float(densities[i] - densities[j])) for i, j in pairs]

    return {
        "overlay_available": 1.0,
        "mask_info_topk_mass": topk_mass,
        "mask_info_topk_iou": topk_iou,
        "mask_info_score": mask_info_score,
        "mask_distinctiveness": pairwise_mean(density_diffs),
    }


def compute_distinctiveness_total(df):
    cols = [
        df.get("geom_distinctiveness", 0.0).fillna(0.0),
        df.get("freq_geo_jsd", 0.0).fillna(0.0),
        df.get("freq_energy_jsd", 0.0).fillna(0.0),
        df.get("mask_distinctiveness", 0.0).fillna(0.0),
    ]
    return 0.4 * cols[0] + 0.25 * cols[1] + 0.15 * cols[2] + 0.20 * cols[3]


def compute_composite(df, weights: ScoreWeights):
    return (
        weights.coverage * df["coverage"].fillna(0.0)
        + weights.low_overlap * (1.0 - df["overlap_ratio"].fillna(1.0))
        + weights.compactness * df["mean_compactness"].fillna(0.0)
        + weights.geom_distinctiveness * df["geom_distinctiveness"].fillna(0.0)
        + weights.size_balance * df["size_balance"].fillna(0.0)
        + weights.region_count * df["region_count_score"].fillna(0.0)
        + weights.mask_info * df["mask_info_score"].fillna(0.0)
        + weights.freq_distinctiveness
        * (0.7 * df["freq_geo_jsd"].fillna(0.0) + 0.3 * df["freq_energy_jsd"].fillna(0.0))
        + weights.mask_distinctiveness * df["mask_distinctiveness"].fillna(0.0)
    )


def pick_overlay_path(
    base_name: str,
    binary_index: Dict[str, str],
    continuous_index: Dict[str, str],
    kind: str,
) -> Optional[str]:
    b = binary_index.get(base_name)
    c = continuous_index.get(base_name)
    if kind == "binary":
        return b
    if kind == "continuous":
        return c
    return c if c is not None else b


def main() -> None:
    args = parse_args()
    if pd is None:
        raise SystemExit(
            "Missing dependency: pandas. Install it (e.g., `pip install pandas`) to run this script."
        )
    if torch is None:
        raise SystemExit(
            "Missing dependency: torch. Install it (e.g., `pip install torch`) to run this script."
        )

    rng = np.random.default_rng(args.seed)
    weights = ScoreWeights(
        coverage=args.w_coverage,
        low_overlap=args.w_low_overlap,
        compactness=args.w_compactness,
        geom_distinctiveness=args.w_geom_distinctiveness,
        size_balance=args.w_size_balance,
        region_count=args.w_region_count,
        mask_info=args.w_mask_info,
        freq_distinctiveness=args.w_freq_distinctiveness,
        mask_distinctiveness=args.w_mask_distinctiveness,
    )

    pairs = collect_mask_files(args.masks_root, args.method_glob)
    if len(pairs) == 0:
        raise SystemExit(
            f"No *_masks.pth found under {args.masks_root} with method-glob={args.method_glob}"
        )

    image_index = build_index(args.image_root, f"*{args.image_ext}")
    binary_index = build_index(args.overlay_root, f"*{args.overlay_suffix_binary}")
    continuous_index = build_index(args.overlay_root, f"*{args.overlay_suffix_continuous}")

    rows = []
    for method, mask_path in pairs:
        base_name = infer_base_name(mask_path, method)
        vocoder = infer_vocoder(base_name, args.vocoder_sep_token)
        masks = load_masks(mask_path)

        geom = geometric_metrics(
            masks=masks,
            target_regions=args.target_regions,
            max_pairs=args.max_pairs,
            rng=rng,
        )

        image_gray = load_image_gray(image_index[base_name]) if base_name in image_index else None
        freq = frequency_metrics(masks=masks, image_gray=image_gray, max_pairs=args.max_pairs, rng=rng)

        overlay_path = pick_overlay_path(
            base_name=base_name,
            binary_index=binary_index,
            continuous_index=continuous_index,
            kind=args.overlay_kind,
        )
        overlay_map = np.load(overlay_path) if overlay_path else None
        ovm = overlay_metrics(
            masks=masks,
            overlay_map=overlay_map,
            topk_fraction=args.topk_fraction,
            max_pairs=args.max_pairs,
            rng=rng,
        )

        row = {
            "method": method,
            "mask_path": mask_path,
            "base_name": base_name,
            "vocoder": vocoder,
            "image_path": image_index.get(base_name),
            "overlay_path": overlay_path,
            "phoneme_purity": np.nan,
            "phoneme_boundary_recall": np.nan,
            "phoneme_entropy": np.nan,
        }
        row.update(geom)
        row.update(freq)
        row.update(ovm)
        rows.append(row)

    per_image = pd.DataFrame(rows)
    per_image["distinctiveness_total"] = compute_distinctiveness_total(per_image)
    per_image["composite_score"] = compute_composite(per_image, weights=weights)

    group_cols = [
        "composite_score",
        "distinctiveness_total",
        "n_regions",
        "coverage",
        "overlap_ratio",
        "mean_compactness",
        "mean_pairwise_iou",
        "geom_distinctiveness",
        "freq_geo_jsd",
        "freq_energy_jsd",
        "mask_info_score",
        "mask_info_topk_mass",
        "mask_info_topk_iou",
        "mask_distinctiveness",
        "size_balance",
        "region_count_score",
        "mean_area",
        "std_area",
        "cv_area",
    ]

    method_summary = (
        per_image.groupby("method")[group_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    method_summary.columns = [
        "method" if c[0] == "method" else f"{c[0]}_{c[1]}" for c in method_summary.columns
    ]
    method_summary = method_summary.sort_values("composite_score_mean", ascending=False)

    feasible = method_summary[
        (method_summary["mask_info_score_mean"].fillna(0.0) >= args.min_mask_info)
        & (
            method_summary["distinctiveness_total_mean"].fillna(0.0)
            >= args.min_total_distinctiveness
        )
    ].copy()
    if len(feasible) > 0:
        recommended = feasible.sort_values(
            ["n_regions_mean", "composite_score_mean"], ascending=[True, False]
        ).iloc[0]
        recommended_method = str(recommended["method"])
    else:
        recommended_method = ""

    if args.output_prefix is None:
        output_prefix = os.path.join(args.masks_root, "region_partition_eval")
    else:
        output_prefix = args.output_prefix

    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    per_image_csv = f"{output_prefix}_per_image.csv"
    summary_csv = f"{output_prefix}_summary.csv"
    per_image.to_csv(per_image_csv, index=False)
    method_summary.to_csv(summary_csv, index=False)

    print(f"Wrote per-image metrics: {per_image_csv}")
    print(f"Wrote method summary: {summary_csv}")
    print("")
    print("Top methods by mean composite_score:")
    for _, row in method_summary.head(10).iterrows():
        print(
            f"{row['method']}: score={row['composite_score_mean']:.4f}, "
            f"regions={row['n_regions_mean']:.2f}, "
            f"mask_info={row['mask_info_score_mean']:.4f}, "
            f"distinct_total={row['distinctiveness_total_mean']:.4f}"
        )

    print("")
    if recommended_method:
        print(
            "Recommended under constraints "
            f"(min_mask_info={args.min_mask_info}, "
            f"min_total_distinctiveness={args.min_total_distinctiveness}): "
            f"{recommended_method}"
        )
    else:
        print(
            "No method met constraints. Relax thresholds or add overlay/image inputs for stronger signals."
        )

    if args.phoneme_align_csv:
        print("")
        print(
            "Note: --phoneme-align-csv was provided, but phoneme metrics are placeholders (NaN) in this version."
        )


if __name__ == "__main__":
    main()
