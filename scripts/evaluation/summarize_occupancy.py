"""Aggregate the history-occupancy ablation across SASRec seeds.

Each input is a JSON report from ``run_occupancy_ablation`` (one dataset,
one seed). The gate/protocol is held fixed; only the SASRec seed varies.
Standard deviations are the sample standard deviation (``ddof=1``),
appropriate for a finite seed replication.

Per dataset x regime x metric (R/N@{5,10,20} and occupancy@{5,10,20}) and
per dataset for the soft-gate weight/penalty summary, we report the mean and
sample std across seeds.

Example:
  python -m scripts.evaluation.summarize_occupancy \
    --inputs outputs/occupancy_{beauty,toys,sports}_seed{0,1,2}.json \
    --out output_final/results/occupancy_ablation/occupancy_ablation_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_reports(paths: list[str]) -> dict[str, dict[int, dict]]:
    """Group reports by dataset name, keyed by seed. Reject duplicates."""
    if len(paths) < 2:
        raise ValueError("at least two per-seed reports are required")
    grouped: dict[str, dict[int, dict]] = {}
    for raw in paths:
        path = Path(raw)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read valid JSON report {path}: {exc}") from exc
        if "regimes" not in report or "dataset" not in report or "seed" not in report:
            raise ValueError(f"{path}: not a run_occupancy_ablation report")
        dataset = str(report["dataset"])
        seed = int(report["seed"])
        if seed in grouped.get(dataset, {}):
            raise ValueError(f"duplicate seed {seed} for dataset {dataset} in {path}")
        grouped.setdefault(dataset, {})[seed] = report
    return grouped


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "values": arr.tolist(),
        "mean": float(arr.mean()),
        "sample_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def summarize_dataset(seed_reports: dict[int, dict]) -> dict:
    """Aggregate one dataset's per-seed reports into mean/std trees."""
    seeds = sorted(seed_reports)
    reference = seed_reports[seeds[0]]
    ref_regimes = reference["regimes"]

    regimes_summary: dict[str, dict] = {}
    for regime, ref_metrics in ref_regimes.items():
        regimes_summary[regime] = {}
        for metric in ref_metrics:
            try:
                vals = [float(seed_reports[s]["regimes"][regime][metric]) for s in seeds]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"all reports must contain regimes/{regime}/{metric}"
                ) from exc
            regimes_summary[regime][metric] = _mean_std(vals)

    # Soft-gate weight/penalty summary (per-expert weights + scalar penalty).
    gate_summary: dict = {}
    if "soft_gate_summary" in reference:
        ref_gate = reference["soft_gate_summary"]
        for expert_idx, expert in enumerate(("seq", "cf", "sem")):
            vals = [float(seed_reports[s]["soft_gate_summary"]["mean_weights"][expert_idx])
                    for s in seeds]
            gate_summary[f"weight_{expert}"] = _mean_std(vals)
        pen_vals = [float(seed_reports[s]["soft_gate_summary"]["mean_penalty"]) for s in seeds]
        gate_summary["penalty"] = _mean_std(pen_vals)

    return {"seeds": seeds, "regimes": regimes_summary, "soft_gate_summary": gate_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Per-seed JSON reports from run_occupancy_ablation")
    parser.add_argument("--out", required=True, help="Output aggregate JSON path")
    args = parser.parse_args()

    grouped = load_reports(args.inputs)
    all_seeds = sorted({s for seeds in grouped.values() for s in seeds})
    summary = {dataset: summarize_dataset(seeds) for dataset, seeds in sorted(grouped.items())}

    aggregate = {
        "protocol": {
            "seeds": all_seeds,
            "mask_history": False,
            "ranking": "full_catalog",
            "gate_fit_split": "validation",
            "standard_deviation": "sample standard deviation (ddof=1)",
        },
        "source_reports": [str(Path(p)) for p in args.inputs],
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
