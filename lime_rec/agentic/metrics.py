"""Per-user and aggregate metrics for agentic conditions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lime_rec.controlled_metrics import evaluate_rankings


def evaluate_agentic(records: Sequence[Mapping], ks=(5, 10)) -> dict[str, float]:
    if not records:
        raise ValueError("records must not be empty")
    rankings = {str(r["user_id"]): ([] if r.get("format_failure") else r["ranking"]) for r in records}
    targets = {str(r["user_id"]): str(r["target_item_id"]) for r in records}
    result = evaluate_rankings(rankings, targets, ks)
    result["candidate_target_coverage"] = sum(
        r["target_item_id"] in r["candidate_ids"] for r in records
    ) / len(records)
    result["format_failure_rate"] = sum(bool(r.get("format_failure")) for r in records) / len(records)

    # Coverage-factorization sanity check. Rankings are pool subsets (format
    # failures are empty), so a target outside the candidate pool can never be
    # hit and overall hit rate must equal coverage x conditional hit rate.
    in_pool = [r for r in records if r["target_item_id"] in r["candidate_ids"]]
    coverage = result["candidate_target_coverage"]
    for k in ks:
        if in_pool:
            conditional = sum(
                str(r["target_item_id"]) in ([] if r.get("format_failure") else r["ranking"])[:k]
                for r in in_pool
            ) / len(in_pool)
        else:
            conditional = 0.0
        result[f"conditional_R@{k}_target_in_pool"] = conditional
        implied = coverage * conditional
        overall = result[f"R@{k}"]
        if abs(overall - implied) > 1e-9:
            raise AssertionError(
                f"R@{k} coverage factorization violated: overall={overall}, "
                f"coverage={coverage}, conditional={conditional}, implied={implied}"
            )
    return result
