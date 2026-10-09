"""Deterministic shared candidate-pool construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidatePool:
    user_id: str
    item_ids: tuple[str, ...]
    candidate_hash: str


def _ordered_top_indices(scores: np.ndarray, item_ids: Sequence[str], n: int) -> list[int]:
    return sorted(range(len(item_ids)), key=lambda i: (-float(scores[i]), str(item_ids[i])))[:n]


def build_candidate_pool(
    user_id: str,
    item_ids: Sequence[str],
    expert_scores: Mapping[str, np.ndarray],
    candidate_per_expert: int = 20,
) -> CandidatePool:
    """Build a stable union. There is deliberately no target argument."""
    if candidate_per_expert < 1:
        raise ValueError("candidate_per_expert must be positive")
    expected = len(item_ids)
    if not expert_scores:
        raise ValueError("expert_scores must not be empty")
    ordered: list[str] = []
    seen: set[str] = set()
    for name in ("sasrec", "itemcf", "semantic"):
        if name not in expert_scores:
            continue
        scores = np.asarray(expert_scores[name])
        if scores.shape != (expected,):
            raise ValueError(f"{name} scores do not align with catalog")
        for index in _ordered_top_indices(scores, item_ids, candidate_per_expert):
            item_id = str(item_ids[index])
            if item_id not in seen:
                seen.add(item_id)
                ordered.append(item_id)
    payload = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False).encode()
    return CandidatePool(user_id, tuple(ordered), hashlib.sha256(payload).hexdigest())
