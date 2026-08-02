"""Summarize any matched three-seed repeat-aware control without paper-specific baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("R@5", "N@5", "R@10", "N@10", "R@20", "N@20")


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    reports = []
    for raw in args.reports:
        path = Path(raw)
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    datasets = {report["dataset"] for report in reports}
    seeds = sorted(report["hyperparameters"]["seed"] for report in reports)
    if len(datasets) != 1 or seeds != [0, 1, 2]:
        raise ValueError("expected one dataset and exactly seeds 0, 1, 2")
    for report in reports:
        if report["protocol"] != {
            "gate_fit_split": "validation",
            "test_used_for_training_or_selection": False,
            "mask_history": False,
            "ranking": "full_catalog",
        }:
            raise ValueError("report does not use the formal gate protocol")
        if report["hyperparameters"].get("initial_weights") != "0.60,0.15,0.25":
            raise ValueError("report does not use the formal initialization")
    reports.sort(key=lambda report: report["hyperparameters"]["seed"])
    out = {
        "label": args.label,
        "dataset": next(iter(datasets)),
        "source_reports": [str(Path(raw)) for raw in args.reports],
        "formal_protocol": reports[0]["protocol"],
        "fusion": {metric: _mean_std([report["test_metrics"][metric] for report in reports])
                   for metric in METRICS},
        "mean_test_weights": {
            name: _mean_std([report["mean_test_weights"][index] for report in reports])
            for index, name in enumerate(("SASRec", "ItemCF", "Semantic"))
        },
        "mean_test_penalty": _mean_std([report["mean_test_penalty"] for report in reports]),
    }
    if all("expert_test_metrics" in report for report in reports):
        out["experts"] = {
            expert: {metric: _mean_std([report["expert_test_metrics"][expert][metric]
                                        for report in reports])
                     for metric in METRICS}
            for expert in ("SASRec", "ItemCF", "Semantic")
        }
        out["fusion_minus_sasrec"] = {
            metric: _mean_std([report["test_metrics"][metric]
                                - report["expert_test_metrics"]["SASRec"][metric]
                                for report in reports])
            for metric in METRICS
        }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
