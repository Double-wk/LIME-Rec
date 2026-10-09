"""Cache-backed, budgeted acquisition of expert evidence.

Policies receive only public views. Labels and uncalled results belong to the
experiment harness. These timings do not measure online expert inference.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import time

import numpy as np

TOOLS = ("sasrec", "itemcf", "semantic")
WEIGHTS = dict(zip(TOOLS, (0.60, 0.15, 0.25)))
SYSTEM = (
    'Select the next recommendation expert to query. sasrec models sequential behavior; '
    'itemcf models collaborative co-occurrence; semantic models frozen item-text similarity. '
    'You see only history and already queried expert results. Select one available tool. '
    'The environment will rank revealed items with fixed weighted score fusion after the '
    'call budget is exhausted. Return only JSON: {"tool":"sasrec"}, substituting an '
    'available tool name. Do not return a ranking or call an unavailable tool.'
)
PROTOCOL = {"version": "budgeted-tools-v1", "weights": WEIGHTS, "history_penalty": 0.10,
            "missing_scores": 0.0, "router_objective": "greedy_next_step_NDCG@10",
            "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()}


class BudgetedEnvironment:
    def __init__(self, record: dict, budget: int):
        if budget not in (1, 2, 3):
            raise ValueError("budget must be 1, 2, or reference-only 3")
        self._record = record
        self.budget = budget
        self.observations: dict = {}

    def view(self) -> dict:
        return copy.deepcopy({"history": self._record["history"],
                              "called_tools": list(self.observations),
                              "available_tools": [t for t in TOOLS if t not in self.observations],
                              "remaining_tool_budget": self.budget - len(self.observations),
                              "observations": self.observations})

    def call(self, tool: str) -> None:
        if len(self.observations) >= self.budget:
            raise ValueError("tool budget exhausted")
        if tool not in TOOLS or tool in self.observations:
            raise ValueError("unknown or duplicate tool")
        self.observations[tool] = copy.deepcopy(self._record["tools"][tool])


def rank_view(view: dict) -> list[str]:
    """Fuse only returned scores; candidates absent from a queried list score zero."""
    observed = view["observations"]
    denominator = sum(WEIGHTS[t] for t in observed)
    scores: dict[str, float] = {}
    for tool, rows in observed.items():
        for row in rows:
            item = row["item_id"]
            scores[item] = scores.get(item, 0.0) + WEIGHTS[tool] * row["score"] / denominator
    history = set(view["history"])
    return sorted(scores, key=lambda i: (-(scores[i] - 0.10 * (i in history)), i))[:10]


def utility(ranking: list[str], target: str) -> float:
    return 1.0 / math.log2(ranking.index(target) + 2) if target in ranking[:10] else 0.0


def features(view: dict) -> np.ndarray:
    history = view["history"]
    out = [math.log1p(len(history)), len(set(history)) / max(1, len(history)),
           float(view["remaining_tool_budget"])]
    for tool in TOOLS:
        rows = view["observations"].get(tool, [])
        values = sorted((r["score"] for r in rows), reverse=True)
        out.extend([float(bool(rows)), values[0] if values else 0.0,
                    values[0] - values[1] if len(values) > 1 else 0.0,
                    float(np.mean(values)) if values else 0.0,
                    sum(r["item_id"] in history for r in rows) / max(1, len(rows))])
    return np.asarray(out, dtype=float)


def messages(view: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(view, sort_keys=True)}]


def training_states(record: dict, target: str):
    for budget in (1, 2):
        prefixes = [()] + ([(t,) for t in TOOLS] if budget == 2 else [])
        for prefix in prefixes:
            env = BudgetedEnvironment(record, budget)
            for tool in prefix:
                env.call(tool)
            view = env.view()
            values = {}
            for tool in view["available_tools"]:
                after = copy.deepcopy(view)
                after["observations"][tool] = record["tools"][tool]
                values[tool] = utility(rank_view(after), target)
            yield view, values


class LinearRouter:
    def __init__(self, state: dict):
        if state["fit_split"] != "validation" or state["protocol"] != PROTOCOL:
            raise ValueError("router must be fitted on validation under this protocol")
        self.state = state

    @classmethod
    def fit(cls, data: dict, ridge: float = 1.0):
        if data["split"] != "validation":
            raise ValueError("router fitting is validation-only")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        states = [(features(v), y) for row in data["rows"]
                  for v, y in training_states(row, data["targets"][row["user_id"]])]
        x = np.stack([v for v, _ in states])
        mean, scale = x.mean(0), x.std(0)
        scale[scale < 1e-8] = 1.0
        x = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        coefficients = {}
        penalty = np.eye(x.shape[1]) * ridge
        penalty[0, 0] = 0
        for tool in TOOLS:
            selected = [i for i, (_, y) in enumerate(states) if tool in y]
            design = x[selected]
            target = np.array([states[i][1][tool] for i in selected])
            coefficients[tool] = np.linalg.solve(design.T @ design + penalty,
                                                  design.T @ target).tolist()
        return cls({"fit_split": "validation", "protocol": PROTOCOL,
                    "identity": data["identity"], "ridge": ridge,
                    "mean": mean.tolist(), "scale": scale.tolist(),
                    "coefficients": coefficients})

    def __call__(self, view: dict) -> str:
        x = np.r_[1.0, (features(view) - self.state["mean"]) / self.state["scale"]]
        return max(view["available_tools"],
                   key=lambda t: float(x @ self.state["coefficients"][t]))


class LLMSelector:
    def __init__(self, client, max_tokens: int = 128):
        self.client = client
        self.max_tokens = max_tokens
        self.events: list[dict] = []

    def __call__(self, view: dict) -> str:
        started = time.perf_counter()
        response = self.client.generate(messages(view), temperature=0.0,
                                        max_tokens=self.max_tokens)
        self.events.append({"latency_seconds": time.perf_counter() - started,
                            "usage": response.get("usage", {}), "text": response["text"]})
        if not isinstance(response["text"], str):
            raise ValueError("expected a JSON text response")
        action = json.loads(response["text"])
        if not isinstance(action, dict) or set(action) != {"tool"} or not isinstance(action["tool"], str):
            raise ValueError("expected exactly one string tool action")
        return action["tool"]


def rollout(record: dict, budget: int, policy) -> dict:
    started = time.perf_counter()
    env = BudgetedEnvironment(record, budget)
    trace, failure = [], None
    for _ in range(budget):
        view = env.view()
        try:
            tool = policy(view)
            env.call(tool)
        except ValueError as exc:
            failure = str(exc)
            trace.append({"view": view, "error": failure})
            break
        trace.append({"view": view, "tool": tool})
    view = env.view()
    return {"user_id": record["user_id"], "ranking": [] if failure else rank_view(view),
            "candidate_ids": sorted({r["item_id"] for rows in view["observations"].values() for r in rows}),
            "tool_calls": view["called_tools"], "budget": budget,
            "format_failure": failure is not None, "failure_reason": failure,
            "trajectory": trace, "cache_replay_wall_seconds": time.perf_counter() - started}


def fixed_policy(sequence):
    return lambda view: sequence[len(view["called_tools"])]


def fixed_sequences(budget: int):
    return itertools.permutations(TOOLS, budget)
