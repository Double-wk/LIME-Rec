"""Summarize matched three-seed repeat-aware gate reports.

The formal summary validates that every input used the same gate initialization
and full-catalog protocol.  It reports the arithmetic mean and sample standard
deviation (ddof=1) for every recorded metric, plus matched fusion--SASRec
differences computed from the exact seed-level reports.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("R@5", "N@5", "R@10", "N@10", "R@20", "N@20")
EXPECTED_PROTOCOL = {
    "gate_fit_split": "validation",
    "test_used_for_training_or_selection": False,
    "mask_history": False,
    "ranking": "full_catalog",
}
EXPECTED_INITIAL_WEIGHTS = (0.60, 0.15, 0.25)
# Published GRAM values used as the fixed in-table comparison context.  These
# are intentionally kept separate from the matched internal expert results.
GRAM_MAIN_TABLE = {
    "amazon_beauty": {"R@5": 0.0641, "N@5": 0.0451, "R@10": 0.0890, "N@10": 0.0531},
    "amazon_toys": {"R@5": 0.0718, "N@5": 0.0516, "R@10": 0.0987, "N@10": 0.0603},
    "amazon_sports": {"R@5": 0.0375, "N@5": 0.0256, "R@10": 0.0554, "N@10": 0.0314},
}


def _mean_std(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("sample standard deviation needs at least two values")
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def _parse_initial_weights(raw: str) -> tuple[float, float, float]:
    weights = tuple(float(value) for value in raw.split(","))
    if len(weights) != 3:
        raise ValueError(f"expected three initial weights, got {raw!r}")
    return weights


def _validate(path: Path, report: dict) -> tuple[str, int]:
    if report.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError(f"{path}: protocol is not the formal full-catalog protocol")
    hyper = report.get("hyperparameters", {})
    initial = _parse_initial_weights(str(hyper.get("initial_weights", "")))
    if not np.allclose(initial, EXPECTED_INITIAL_WEIGHTS, rtol=0, atol=1e-12):
        raise ValueError(f"{path}: initial weights {initial} are not {EXPECTED_INITIAL_WEIGHTS}")
    if Path(str(hyper.get("semantic_emb", ""))).name != "amazon_" + report["dataset"].removeprefix("amazon_") + "_bge_base.npz":
        raise ValueError(f"{path}: main table requires the BGE embedding artifact")
    seed = hyper.get("seed")
    if not isinstance(seed, int):
        raise ValueError(f"{path}: missing integer seed")
    if not isinstance(report.get("expert_test_metrics"), dict):
        raise ValueError(f"{path}: missing expert_test_metrics; run annotate_repeat_aware_expert_metrics first")
    return str(report["dataset"]), seed


def _metric_summary(reports: list[dict], key: str) -> dict[str, dict[str, float]]:
    return {
        metric: _mean_std([float(report[key][metric]) for report in reports])
        for metric in METRICS
    }


def _summarize_dataset(reports: list[dict]) -> dict:
    reports = sorted(reports, key=lambda report: report["hyperparameters"]["seed"])
    fusion = _metric_summary(reports, "test_metrics")
    experts = {
        expert: {
            metric: _mean_std([float(report["expert_test_metrics"][expert][metric]) for report in reports])
            for metric in METRICS
        }
        for expert in ("SASRec", "ItemCF", "Semantic")
    }
    lifts = {
        metric: _mean_std([
            float(report["test_metrics"][metric])
            - float(report["expert_test_metrics"]["SASRec"][metric])
            for report in reports
        ])
        for metric in METRICS
    }
    r10_lift = lifts["R@10"]["mean"]
    sasrec_r10 = experts["SASRec"]["R@10"]["mean"]
    gram_relative_lifts = {
        metric: 100.0 * (fusion[metric]["mean"] / baseline - 1.0)
        for metric, baseline in GRAM_MAIN_TABLE[reports[0]["dataset"]].items()
    }
    return {
        "seeds": [report["hyperparameters"]["seed"] for report in reports],
        "fusion": fusion,
        "experts": experts,
        "fusion_minus_sasrec": lifts,
        "r10_relative_lift_vs_sasrec_percent": 100.0 * r10_lift / sasrec_r10,
        "relative_lift_vs_gram_percent": gram_relative_lifts,
        "mean_test_weights": {
            name: _mean_std([float(report["mean_test_weights"][index]) for report in reports])
            for index, name in enumerate(("SASRec", "ItemCF", "Semantic"))
        },
        "mean_test_penalty": _mean_std([float(report["mean_test_penalty"]) for report in reports]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    sources: list[str] = []
    seen: set[tuple[str, int]] = set()
    for raw_path in args.reports:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as fh:
            report = json.load(fh)
        dataset, seed = _validate(path, report)
        key = (dataset, seed)
        if key in seen:
            raise ValueError(f"duplicate report for {dataset}, seed {seed}")
        seen.add(key)
        grouped[dataset].append(report)
        sources.append(str(path))

    expected_datasets = {"amazon_beauty", "amazon_toys", "amazon_sports"}
    if set(grouped) != expected_datasets:
        raise ValueError(f"expected datasets {sorted(expected_datasets)}, got {sorted(grouped)}")
    for dataset, reports in grouped.items():
        seeds = sorted(report["hyperparameters"]["seed"] for report in reports)
        if seeds != [0, 1, 2]:
            raise ValueError(f"{dataset}: expected seeds [0, 1, 2], got {seeds}")

    output = {
        "source_reports": sorted(sources),
        "formal_protocol": {
            **EXPECTED_PROTOCOL,
            "initial_weights": list(EXPECTED_INITIAL_WEIGHTS),
            "encoder": "BGE base",
            "statistic": "mean and sample standard deviation across seeds 0, 1, 2",
        },
        "summary": {dataset: _summarize_dataset(reports) for dataset, reports in sorted(grouped.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
