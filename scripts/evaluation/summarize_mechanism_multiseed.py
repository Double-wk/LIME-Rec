"""Summarize formal repeat-aware mechanism exports across matched seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.evaluation.analyze_complementarity import analyze


EXPECTED_PROTOCOL = {
    "gate_fit_split": "validation",
    "test_used_for_training_or_selection": False,
    "mask_history": False,
    "ranking": "full_catalog",
}


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean_std(values: list[float]) -> dict[str, float]:
    values_np = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values_np.mean()),
        "std": float(values_np.std(ddof=1)) if len(values_np) > 1 else 0.0,
    }


def _summarize_reports(reports: list[dict]) -> dict:
    def collect(section: str) -> dict[str, dict[str, float]]:
        keys = reports[0]["analysis"][section]
        return {key: _mean_std([report["analysis"][section][key] for report in reports])
                for key in keys}

    lift = [report["analysis"]["rq3_3_lift_attribution"] for report in reports]
    return {
        "jaccard_at_10": collect("rq3_1_jaccard_mean"),
        "full_catalog_spearman": collect("rq3_2_spearman_mean"),
        "hit_set_decomposition_percent": collect("rq3_4_hit_set_decomposition_pct"),
        "recovered_user_rate_percent": _mean_std(
            [100.0 * report["recovered_pct"] for report in lift]
        ),
        "fusion_hit10": _mean_std([report["hit10"]["Fusion"] for report in lift]),
        "expert_hit10": {
            expert: _mean_std([report["hit10"][expert] for report in lift])
            for expert in ("SASRec", "ItemCF", "Semantic")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reports = []
    for raw in args.inputs:
        path = Path(raw)
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("protocol") != EXPECTED_PROTOCOL:
            raise ValueError(f"{path}: does not use the formal no-mask protocol")
        gate = metadata.get("gate", {})
        if gate.get("initial_weights") != [0.60, 0.15, 0.25]:
            raise ValueError(f"{path}: gate initialization is not [0.60, 0.15, 0.25]")
        rows = _read_rows(path)
        analysis = analyze(rows, K_jacc=metadata["sample"]["topk"], K_lift=10)
        reports.append({
            "input": str(path),
            "metadata": metadata,
            "analysis": analysis,
            "recovered_pct": analysis["rq3_3_lift_attribution"]["recovered_pct"],
            "hit10": analysis["rq3_3_lift_attribution"]["hit10"],
        })

    datasets = {report["metadata"]["dataset"] for report in reports}
    if len(datasets) != 1:
        raise ValueError(f"expected one dataset per summary, got {sorted(datasets)}")
    seeds = sorted(report["metadata"]["gate"]["seed"] for report in reports)
    if seeds != [0, 1, 2]:
        raise ValueError(f"expected matched seeds [0, 1, 2], got {seeds}")

    out = {
        "dataset": next(iter(datasets)),
        "formal_protocol": {
            **EXPECTED_PROTOCOL,
            "initial_weights": [0.60, 0.15, 0.25],
            "statistic": "mean and sample standard deviation across seeds 0, 1, 2",
        },
        "per_seed": sorted(reports, key=lambda report: report["metadata"]["gate"]["seed"]),
        "summary": _summarize_reports(reports),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
