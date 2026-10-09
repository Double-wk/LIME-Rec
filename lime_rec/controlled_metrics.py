"""Shared single-target rank-list metrics for controlled comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    targets: Mapping[str, str],
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Evaluate ordered raw-item rankings without target injection or user dropping."""
    if set(rankings) != set(targets):
        missing = sorted(set(targets) - set(rankings))
        extra = sorted(set(rankings) - set(targets))
        raise ValueError(f"ranking/target users differ: missing={missing[:5]} extra={extra[:5]}")
    if not rankings:
        raise ValueError("rankings must not be empty")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")

    totals = {f"R@{k}": 0.0 for k in ks}
    totals.update({f"N@{k}": 0.0 for k in ks})
    for user_id in sorted(targets):
        ranking = list(rankings[user_id])
        target = targets[user_id]
        try:
            rank = ranking.index(target)
        except ValueError:
            rank = None
        for k in ks:
            hit = rank is not None and rank < k
            totals[f"R@{k}"] += float(hit)
            totals[f"N@{k}"] += 1.0 / math.log2(rank + 2) if hit else 0.0
    return {name: value / len(targets) for name, value in totals.items()}


def stable_unique_ranking(items: Sequence[str]) -> tuple[list[str], int]:
    """Stably deduplicate predictions and report how many duplicates occurred."""
    seen: set[str] = set()
    result: list[str] = []
    duplicates = 0
    for item in items:
        if item in seen:
            duplicates += 1
            continue
        seen.add(item)
        result.append(item)
    return result, duplicates
