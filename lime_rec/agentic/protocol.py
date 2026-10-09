"""Agent state, validation, and fail-closed cross-condition assertions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class AgentCondition(str, Enum):
    RECOVERY = "candidate_recovery"
    ALL_TOOLS = "llm_all_tools"
    ADAPTIVE = "adaptive_agent"
    NO_SEMANTIC = "adaptive_agent_no_semantic"


@dataclass
class AgentState:
    user_id_hash: str
    history_length: int
    candidate_ids: tuple[str, ...]
    history_membership: tuple[bool, ...]
    called_tools: list[str] = field(default_factory=list)
    observations: dict[str, dict] = field(default_factory=dict)
    remaining_budget: int = 3
    messages: list[dict] = field(default_factory=list)
    final_ranking: list[str] | None = None

    @classmethod
    def create(cls, user_id: str, history: Sequence[str], candidates: Sequence[str], budget: int = 3):
        if budget < 0:
            raise ValueError("budget must be non-negative")
        history_set = set(history)
        return cls(hashlib.sha256(user_id.encode()).hexdigest(), len(history), tuple(candidates),
                   tuple(item in history_set for item in candidates), remaining_budget=budget)


def validate_action(action: Mapping, state: AgentState, allowed_tools: Sequence[str]) -> None:
    if action.get("action") == "call_tool":
        tool = action.get("tool")
        if tool not in allowed_tools:
            raise ValueError("agent requested unknown or disallowed tool")
        if tool in state.called_tools:
            raise ValueError("agent cannot repeat a tool")
        if state.remaining_budget <= 0:
            raise ValueError("agent tool budget exhausted")
    elif action.get("action") == "finish":
        validate_ranking(action.get("ranking"), state.candidate_ids)
    else:
        raise ValueError("agent action must be call_tool or finish")


def validate_ranking(ranking: object, candidates: Sequence[str], min_length: int = 10) -> None:
    if not isinstance(ranking, list) or len(ranking) < min_length:
        raise ValueError(f"ranking must contain at least {min_length} items")
    if len(ranking) != len(set(ranking)):
        raise ValueError("ranking contains duplicate items")
    if not set(ranking).issubset(set(candidates)):
        raise ValueError("ranking must be a subset of the candidate pool")


def _assert_equal(label: str, values: Sequence[object]) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise RuntimeError(f"protocol mismatch: {label}")


def assert_same_users(records_by_condition: Mapping[str, Sequence[Mapping]]) -> None:
    _assert_equal("users", [tuple(row["user_id"] for row in rows) for rows in records_by_condition.values()])


def assert_same_targets(records_by_condition: Mapping[str, Sequence[Mapping]]) -> None:
    _assert_equal("targets", [tuple(row["target_item_id"] for row in rows) for rows in records_by_condition.values()])


def assert_same_candidate_pool(records_by_condition: Mapping[str, Sequence[Mapping]]) -> None:
    _assert_equal("candidate pools", [tuple(row["candidate_hash"] for row in rows) for rows in records_by_condition.values()])


def assert_same_expert_scores(records_by_condition: Mapping[str, Sequence[Mapping]]) -> None:
    _assert_equal("expert scores", [tuple(row["expert_score_hashes"] for row in rows) for rows in records_by_condition.values()])
