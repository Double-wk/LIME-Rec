"""Bootstrap 95% CI for R@K / NDCG@K from per-user metrics JSONL.

Reads outputs/per_user_metrics_{dataset}.jsonl. For each method (SASRec /
ItemCF / Semantic / Fusion), computes per-user hit@K and NDCG@K from the
stored `target_rank`, then resamples test users with replacement
n_resamples times to produce 95% percentile confidence intervals.

Also reports a paired bootstrap of (Fusion − SASRec) lift, where every
resampled user contributes (fusion_hit − sas_hit) → distribution of the
lift, 95% CI.

Usage:
  python3 -m scripts.evaluation.compute_bootstrap_ci \
    --input outputs/per_user_metrics_amazon_beauty.jsonl \
    --K 10 --n-resamples 1000 \
    --out-json outputs/bootstrap_ci_amazon_beauty.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("SASRec", "ItemCF", "Semantic", "Fusion")


def hit_array(entries, method, K):
    out = np.empty(len(entries), dtype=np.float64)
    for i, e in enumerate(entries):
        if method == "Fusion":
            fkey = list(e["fusion"].keys())[0]
            r = e["fusion"][fkey]["target_rank"]
        else:
            r = e["experts"][method]["target_rank"]
        out[i] = 1.0 if r < K else 0.0
    return out


def ndcg_array(entries, method, K):
    out = np.empty(len(entries), dtype=np.float64)
    for i, e in enumerate(entries):
        if method == "Fusion":
            fkey = list(e["fusion"].keys())[0]
            r = e["fusion"][fkey]["target_rank"]
        else:
            r = e["experts"][method]["target_rank"]
        out[i] = 1.0 / math.log2(r + 2) if r < K else 0.0
    return out


def bootstrap_ci(values: np.ndarray, n_resamples: int, alpha: float = 0.05,
                 rng: np.random.Generator | None = None):
    """Percentile bootstrap CI for the mean of `values`.

    Returns (point_estimate, lo, hi) for confidence level (1 - alpha).
    """
    rng = rng or np.random.default_rng(0)
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))  # shape (R, N)
    boot_means = values[idx].mean(axis=1)            # shape (R,)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


def paired_bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray,
                        n_resamples: int, alpha: float = 0.05,
                        rng: np.random.Generator | None = None):
    """Paired bootstrap CI for mean(a) − mean(b), resampling indices jointly."""
    rng = rng or np.random.default_rng(0)
    n = len(values_a)
    diffs = values_a - values_b
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(diffs.mean()), lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"[loaded] {args.input} ({len(entries)} users)")

    rng = np.random.default_rng(args.seed)
    K = args.K
    R = args.n_resamples

    # Compute per-method per-user hit and NDCG
    hits = {m: hit_array(entries, m, K) for m in METHODS}
    ndcg = {m: ndcg_array(entries, m, K) for m in METHODS}

    out = {
        "dataset_input": args.input,
        "n_users": len(entries),
        "K": K,
        "n_resamples": R,
        "seed": args.seed,
        "metrics": {},
    }

    print(f"\n{'Method':<10}{'metric':<8}{'point':>9}{'lo (2.5%)':>12}{'hi (97.5%)':>12}")
    print("-" * 51)
    for m in METHODS:
        for metric_name, arr in (("R@K", hits[m]), ("N@K", ndcg[m])):
            point, lo, hi = bootstrap_ci(arr, R, rng=rng)
            out["metrics"].setdefault(m, {})[metric_name] = {
                "point": point, "lo": lo, "hi": hi,
            }
            label_metric = metric_name.replace("K", str(K))
            print(f"{m:<10}{label_metric:<8}{point:>9.4f}{lo:>12.4f}{hi:>12.4f}")

    # Paired (Fusion − SASRec) lift CIs
    print(f"\n{'pair':<24}{'metric':<8}{'point':>9}{'lo (2.5%)':>12}{'hi (97.5%)':>12}")
    print("-" * 65)
    paired = {}
    for metric_name, hits_or_ndcg in (("R@K", hits), ("N@K", ndcg)):
        a = hits_or_ndcg["Fusion"]
        b = hits_or_ndcg["SASRec"]
        point, lo, hi = paired_bootstrap_ci(a, b, R, rng=rng)
        paired.setdefault("Fusion-SASRec", {})[metric_name] = {
            "point": point, "lo": lo, "hi": hi,
        }
        label_metric = metric_name.replace("K", str(K))
        print(f"{'Fusion - SASRec':<24}{label_metric:<8}"
              f"{point:>+9.4f}{lo:>+12.4f}{hi:>+12.4f}")
    out["paired_lift"] = paired

    # Flag significance: lift CI does not overlap zero
    for pair, mets in paired.items():
        for mn, d in mets.items():
            sig = (d["lo"] > 0) or (d["hi"] < 0)
            d["significant_at_95"] = sig

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
