"""Aggregate the calibration/fusion isolation ablation over three seeds.

Reads the per-seed reports written by run_repeat_aware_gate.py under
output_final/results/tab_isolation and prints a compact R@10/N@10 table with
the seed mean and sample standard deviation for each of the four configurations:

  1. sasrec_nocal   -- bare SASRec, no history calibration
  2. sasrec_cal     -- SASRec + bounded history calibration
  3. fusion_nocal   -- three-expert fusion, no calibration
  4. fusion_full    -- full LIME-Rec (fusion + calibration)

The two contrasts that isolate the calibration contribution from the
semantic-fusion contribution are also reported:
  calibration gain  = sasrec_cal   - sasrec_nocal
  fusion gain       = fusion_nocal - sasrec_nocal
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("output_final/results/tab_isolation")
CONFIGS = ["sasrec_nocal", "sasrec_cal", "fusion_nocal", "fusion_full"]
DATASETS = ["beauty", "toys", "sports"]
SEEDS = [0, 1, 2]
METRICS = ["R@5", "N@5", "R@10", "N@10"]


def load(config: str, dataset: str, seed: int) -> dict:
    path = OUT / f"{config}_{dataset}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))["test_metrics"]


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    summary: dict = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for config in CONFIGS:
            per_metric = {}
            for metric in METRICS:
                values = [load(config, dataset, seed)[metric] for seed in SEEDS]
                mean, std = mean_std(values)
                per_metric[metric] = {"mean": mean, "std": std}
            summary[dataset][config] = per_metric

    # Console table (R@10 focus).
    header = f"{'dataset':<8}{'sasrec_nocal':>16}{'sasrec_cal':>16}{'fusion_nocal':>16}{'fusion_full':>16}"
    print("=== R@10 (mean +/- std over seeds 0,1,2) ===")
    print(header)
    for dataset in DATASETS:
        cells = []
        for config in CONFIGS:
            entry = summary[dataset][config]["R@10"]
            cells.append(f"{entry['mean']:.4f}+/-{entry['std']:.4f}")
        print(f"{dataset:<8}" + "".join(f"{c:>16}" for c in cells))

    print("\n=== Contribution decomposition (R@10, seed mean) ===")
    print(f"{'dataset':<8}{'calib gain':>14}{'fusion gain':>14}{'full gain':>14}{'total':>14}")
    for dataset in DATASETS:
        base = summary[dataset]["sasrec_nocal"]["R@10"]["mean"]
        cal = summary[dataset]["sasrec_cal"]["R@10"]["mean"]
        fus = summary[dataset]["fusion_nocal"]["R@10"]["mean"]
        full = summary[dataset]["fusion_full"]["R@10"]["mean"]
        print(f"{dataset:<8}{cal - base:>+14.4f}{fus - base:>+14.4f}"
              f"{full - fus:>+14.4f}{full - base:>+14.4f}")

    out_path = OUT / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
