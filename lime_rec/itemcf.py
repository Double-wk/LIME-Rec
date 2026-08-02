"""Deterministic ItemCF model construction.

The public ItemCF artifact deliberately contains only statistics derived from
the training portion of each user's history.  Validation and test targets are
never read by this module.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Mapping, Sequence


MODEL_FORMAT_VERSION = 1


def build_itemcf_model(
    *,
    dataset_name: str,
    histories: Mapping[str, Sequence[str]],
    item_ids: Sequence[str],
    top_k: int = 100,
    window_size: int = 20,
) -> dict:
    """Build a sparse, cosine-normalized item co-occurrence model.

    Every pair of distinct items that appears within ``window_size`` positions
    in a *training* history contributes one symmetric co-occurrence.  The
    resulting score is ``cooccurrence / sqrt(freq_i * freq_j)``.  Keeping the
    top neighbours per source item gives a compact JSON artifact that the
    :class:`lime_rec.experts.ItemCFExpert` can load directly.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if window_size < 1:
        raise ValueError("window_size must be positive")

    frequencies: Counter[str] = Counter()
    cooccurrences: dict[str, Counter[str]] = defaultdict(Counter)
    interaction_count = 0

    for history in histories.values():
        sequence = [str(item_id) for item_id in history if item_id]
        frequencies.update(sequence)
        interaction_count += len(sequence)
        for right_index, right_item in enumerate(sequence):
            left_start = max(0, right_index - window_size)
            for left_item in sequence[left_start:right_index]:
                if left_item == right_item:
                    continue
                cooccurrences[left_item][right_item] += 1
                cooccurrences[right_item][left_item] += 1

    neighbours: dict[str, list[dict[str, float | str]]] = {}
    for source_item in sorted(cooccurrences):
        source_frequency = frequencies[source_item]
        scored = [
            (
                target_item,
                count / math.sqrt(source_frequency * frequencies[target_item]),
            )
            for target_item, count in cooccurrences[source_item].items()
            if frequencies[target_item]
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        neighbours[source_item] = [
            {"item_id": target_item, "score": score}
            for target_item, score in scored[:top_k]
        ]

    return {
        "format_version": MODEL_FORMAT_VERSION,
        "dataset": dataset_name,
        "training_only": True,
        "normalization": "cooccurrence / sqrt(item_frequency_i * item_frequency_j)",
        "window_size": window_size,
        "scoring_history_length": window_size,
        "top_k": top_k,
        "num_training_users": len(histories),
        "num_training_interactions": interaction_count,
        "num_catalog_items": len(item_ids),
        "num_items_with_neighbours": len(neighbours),
        "neighbors": neighbours,
    }
