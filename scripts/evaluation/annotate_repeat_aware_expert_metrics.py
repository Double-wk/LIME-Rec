"""Add matched singleton-expert test metrics to repeat-aware gate reports.

The repeat-aware gate runner records fusion metrics by default.  This utility
reconstructs the three frozen expert score matrices under precisely the report
configuration and adds their full-catalog test metrics in place.  It lets the
formal multi-seed summary derive fusion--SASRec differences from the same nine
reports rather than from a separate, potentially stale baseline artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lime_rec.evaluation import ExpertEvaluator
from scripts.evaluation.run_repeat_aware_gate import batch_scores, collect_examples


EXPERTS = ("SASRec", "ItemCF", "Semantic")


def _load_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        report = json.load(fh)
    hyper = report.get("hyperparameters", {})
    required = ("config", "sasrec_model", "itemcf_model", "semantic_emb", "device")
    missing = [key for key in required if not hyper.get(key)]
    if missing:
        raise ValueError(f"{path}: missing hyperparameter fields {missing}")
    protocol = report.get("protocol", {})
    expected = {
        "gate_fit_split": "validation",
        "test_used_for_training_or_selection": False,
        "mask_history": False,
        "ranking": "full_catalog",
    }
    if protocol != expected:
        raise ValueError(f"{path}: unexpected protocol {protocol}")
    return report


def _metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    """Compute the gate-runner metrics without retaining full score matrices."""
    result: dict[str, float] = {}
    for k in (5, 10, 20):
        hits = ranks < k
        result[f"R@{k}"] = float(hits.mean())
        result[f"N@{k}"] = float(np.where(hits, 1.0 / np.log2(ranks + 2), 0.0).mean())
    return result


def _expert_metrics(report: dict, batch_size: int, device: str) -> dict[str, dict[str, float]]:
    hyper = report["hyperparameters"]
    evaluator = ExpertEvaluator.from_paths(
        hyper["config"], hyper["sasrec_model"], hyper["itemcf_model"], hyper["semantic_emb"],
        device=device, mask_history=False,
    )
    test = collect_examples(evaluator, "test")
    all_ranks = [[] for _ in EXPERTS]
    for start in tqdm(range(0, len(test), batch_size), desc=f"experts-{report['dataset']}", unit="batch"):
        scores, _, targets, _ = batch_scores(evaluator, test[start:start + batch_size], device)
        scores_np = scores.cpu().numpy()
        targets_np = targets.cpu().numpy()
        for index in range(len(EXPERTS)):
            expert_scores = scores_np[:, index, :]
            target_scores = expert_scores[np.arange(len(targets_np)), targets_np]
            all_ranks[index].append((expert_scores > target_scores[:, None]).sum(axis=1))
    return {
        name: _metrics_from_ranks(np.concatenate(all_ranks[index]))
        for index, name in enumerate(EXPERTS)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True,
                        help="Repeat-aware gate JSON reports to enrich in place.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=None,
        help=("Device for the newly computed expert metrics. Defaults to each "
              "report's recorded device; use an explicit override to preserve "
              "the provenance of a new computation."),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    for raw_path in args.reports:
        path = Path(raw_path)
        report = _load_report(path)
        source_device = report["hyperparameters"]["device"]
        device = args.device or source_device
        report["expert_test_metrics"] = _expert_metrics(report, args.batch_size, device)
        report["expert_metrics_protocol"] = {
            "split": "test",
            "mask_history": False,
            "ranking": "full_catalog",
            "experts": list(EXPERTS),
            "device": device,
            "source_report_device": source_device,
        }
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
