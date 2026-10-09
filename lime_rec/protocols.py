"""Protocol helpers shared by controlled and agentic experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .data import Dataset


def truncate_history(history: Sequence[str], history_cap: int | None = None) -> list[str]:
    """Return a copy containing the most recent ``history_cap`` interactions."""
    values = list(history)
    if history_cap is None:
        return values
    if history_cap < 1:
        raise ValueError("history_cap must be positive or None")
    return values[-history_cap:]


@dataclass(frozen=True)
class PredictionContext:
    """Information available before a recommendation is produced."""

    user_id: str
    history: tuple[str, ...]
    split: str


@dataclass(frozen=True)
class EvaluationLabel:
    """Held-out information available only to evaluation code."""

    user_id: str
    target_item_id: str
    split: str


def prediction_context(dataset: Dataset, user_id: str, split: str) -> PredictionContext:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    history = list(dataset.history_by_user[user_id])
    if split == "test":
        history.append(dataset.valid_by_user[user_id])
    return PredictionContext(user_id=user_id, history=tuple(history), split=split)


def evaluation_label(dataset: Dataset, user_id: str, split: str) -> EvaluationLabel:
    if split == "validation":
        target = dataset.valid_by_user[user_id]
    elif split == "test":
        target = dataset.test_by_user[user_id]
    else:
        raise ValueError("split must be validation or test")
    return EvaluationLabel(user_id=user_id, target_item_id=target, split=split)
