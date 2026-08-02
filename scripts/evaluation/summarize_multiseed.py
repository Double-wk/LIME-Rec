"""Aggregate full-catalog shared-gate evaluations across SASRec seeds.

Each input must be a JSON report produced by ``run_shared_weight``.  The
semantic embedding, ItemCF model, gate, and evaluation protocol are held fixed;
only the SASRec initialization/data-order seed varies.  Standard deviations are
the sample standard deviation (``ddof=1``), appropriate for a finite seed
replication rather than a population estimate.

Example:
  python -m scripts.evaluation.summarize_multiseed \
    --inputs output/results/supplementary/multiseed/multiseed_shared_weight_bge_base_seed{0,1,2}.json \
    --out output/results/supplementary/multiseed/multiseed_shared_weight_bge_base_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _seed_from_report(report: dict, source: Path) -> int:
    """Read the training seed encoded by run_shared_weight's SASRec suffix."""
    suffix = str(report.get("sasrec_suffix", ""))
    match = re.fullmatch(r"seed(\d+)", suffix)
    if not match:
        raise ValueError(
            f"{source}: expected sasrec_suffix of the form 'seedN', got {suffix!r}"
        )
    return int(match.group(1))


def load_reports(paths: list[str]) -> tuple[list[tuple[int, Path, dict]], dict, str]:
    """Load compatible reports and return seed-sorted reports plus shared metadata."""
    if len(paths) < 2:
        raise ValueError("at least two per-seed reports are required")

    loaded: list[tuple[int, Path, dict]] = []
    expected_weights: dict | None = None
    expected_encoder: str | None = None
    seen_seeds: set[int] = set()
    for raw_path in paths:
        path = Path(raw_path)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read valid JSON report {path}: {exc}") from exc

        weights = report.get("shared_weights")
        encoder = report.get("encoder")
        results = report.get("results")
        if not isinstance(weights, dict) or not isinstance(encoder, str) or not isinstance(results, dict):
            raise ValueError(f"{path}: not a run_shared_weight report")
        if expected_weights is None:
            expected_weights, expected_encoder = weights, encoder
        elif weights != expected_weights or encoder != expected_encoder:
            raise ValueError(
                f"{path}: gate/encoder differs from the first report; seed summaries "
                "must use one fixed evaluation protocol"
            )

        seed = _seed_from_report(report, path)
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed {seed} in {path}")
        seen_seeds.add(seed)
        loaded.append((seed, path, results))

    return sorted(loaded), expected_weights or {}, expected_encoder or ""


def summarize(loaded: list[tuple[int, Path, dict]]) -> dict:
    """Compute per-dataset, per-method metric means and sample standard deviations."""
    reference = loaded[0][2]
    datasets = list(reference)
    if not datasets:
        raise ValueError("the first report contains no datasets")

    summary: dict[str, dict] = {}
    for dataset in datasets:
        methods = reference[dataset]
        if not isinstance(methods, dict):
            raise ValueError(f"invalid result structure for dataset {dataset!r}")
        summary[dataset] = {}
        for method, first_metrics in methods.items():
            if not isinstance(first_metrics, dict):
                raise ValueError(f"invalid metrics for {dataset}/{method}")
            summary[dataset][method] = {}
            for metric in first_metrics:
                try:
                    values = [float(results[dataset][method][metric]) for _, _, results in loaded]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"all reports must contain {dataset}/{method}/{metric}"
                    ) from exc
                summary[dataset][method][metric] = {
                    "values": values,
                    "mean": float(np.mean(values)),
                    "sample_std": float(np.std(values, ddof=1)),
                }

    # Reject a report that contains a differently shaped dataset/method/metric tree.
    expected_tree = {
        dataset: {method: set(metrics) for method, metrics in methods.items()}
        for dataset, methods in reference.items()
    }
    for _, path, results in loaded[1:]:
        tree = {
            dataset: {method: set(metrics) for method, metrics in methods.items()}
            for dataset, methods in results.items()
            if isinstance(methods, dict)
        }
        if tree != expected_tree:
            raise ValueError(f"{path}: dataset/method/metric structure differs from the first report")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Per-seed JSON reports from run_shared_weight")
    parser.add_argument("--out", required=True, help="Output aggregate JSON path")
    args = parser.parse_args()

    loaded, weights, encoder = load_reports(args.inputs)
    aggregate = {
        "protocol": {
            "seeds": [seed for seed, _, _ in loaded],
            "shared_weights": weights,
            "encoder": encoder,
            "semantic_and_itemcf": "fixed across seeds",
            "standard_deviation": "sample standard deviation (ddof=1)",
        },
        "source_reports": [str(path) for _, path, _ in loaded],
        "per_seed": {str(seed): results for seed, _, results in loaded},
        "summary": summarize(loaded),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
