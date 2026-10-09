"""Evaluate either controlled model through the shared rank-list evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lime_rec.controlled_metrics import evaluate_rankings
from lime_rec.evaluation import load_configured_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    alignment = json.loads(Path(args.alignment).read_text(encoding="utf-8"))
    if not alignment.get("passed"):
        raise RuntimeError("controlled evaluation blocked by failed alignment audit")
    dataset = load_configured_dataset(args.config)
    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line]
    by_user = {row["user_id"]: row for row in rows}
    if set(by_user) != set(dataset.user_ids) or len(by_user) != len(rows):
        raise RuntimeError("predictions must contain each dataset user exactly once")
    for user_id, row in by_user.items():
        if row["target_item_id"] != dataset.test_by_user[user_id]:
            raise RuntimeError(f"target mismatch for {user_id}")
        if not set(row["ranking"]).issubset(set(dataset.item_ids)):
            raise RuntimeError(f"out-of-catalog prediction for {user_id}")
    metrics = evaluate_rankings({u: row["ranking"] for u, row in by_user.items()},
                                dataset.test_by_user, ks=(5, 10))
    result = {"dataset": dataset.name, "model": rows[0]["model"], "seed": rows[0]["seed"],
              "metrics": metrics, "alignment_passed": True,
              "prediction_path": args.predictions}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
