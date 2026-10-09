"""User-level paired bootstrap for frozen agentic conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {row["user_id"]: row for row in rows}


def _values(rows: dict[str, dict], users: list[str], metric: str) -> np.ndarray:
    k = int(metric.split("@")[1])
    values = []
    for user in users:
        rank = rows[user]["target_rank"]
        if metric.startswith("R"):
            values.append(float(rank is not None and rank < k))
        else:
            values.append(1.0 / np.log2(rank + 2) if rank is not None and rank < k else 0.0)
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    root = Path(args.dir)
    conditions = {name: _load(root / f"{name}.jsonl") for name in
                  ("recovery", "llm_all_tools", "adaptive_agent")}
    user_sets = [set(rows) for rows in conditions.values()]
    if any(users != user_sets[0] for users in user_sets[1:]):
        raise RuntimeError("paired bootstrap requires identical users")
    users = sorted(user_sets[0])
    rng = np.random.default_rng(args.seed)
    comparisons = [("adaptive_agent", "recovery"), ("llm_all_tools", "recovery"),
                   ("adaptive_agent", "llm_all_tools")]
    output = {"n_resamples": args.n_resamples, "seed": args.seed,
              "resampling_unit": "user", "comparisons": {}}
    for left, right in comparisons:
        key = f"{left} - {right}"
        output["comparisons"][key] = {}
        for metric in ("R@10", "N@10"):
            diff = _values(conditions[left], users, metric) - _values(conditions[right], users, metric)
            estimates = np.empty(args.n_resamples)
            for index in range(args.n_resamples):
                estimates[index] = diff[rng.integers(0, len(diff), len(diff))].mean()
            output["comparisons"][key][metric] = {
                "point_estimate": float(diff.mean()),
                "ci95_percentile": np.percentile(estimates, [2.5, 97.5]).tolist()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
