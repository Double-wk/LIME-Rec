"""Summarize utility, reliability, and measured serving costs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from lime_rec.agentic.metrics import evaluate_agentic
from lime_rec.agentic.protocol import (assert_same_candidate_pool, assert_same_expert_scores,
                                       assert_same_targets, assert_same_users)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()
    root = Path(args.dir)
    names = ("recovery", "llm_all_tools", "adaptive_agent", "adaptive_agent_no_semantic")
    records = {name: [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines() if line]
               for name in names}
    main = {name: records[name] for name in names[:3]}
    assert_same_users(main); assert_same_targets(main)
    assert_same_candidate_pool(main); assert_same_expert_scores(main)
    rows = []
    for name, values in records.items():
        metrics = evaluate_agentic(values)
        latencies = np.asarray([row.get("latency_seconds", 0.0) for row in values])
        row = {"condition": name, **metrics,
               "mean_llm_calls_per_user": float(np.mean([r.get("llm_calls", 0) for r in values])),
               "mean_tool_calls_per_user": float(np.mean([len(r.get("tool_calls", [])) for r in values])),
               "mean_input_tokens": float(np.mean([r.get("input_tokens", 0) for r in values])),
               "mean_output_tokens": float(np.mean([r.get("output_tokens", 0) for r in values])),
               "retry_rate": float(np.mean([r.get("retry_count", 0) > 0 for r in values])),
               "median_latency_seconds": float(np.median(latencies)),
               "mean_latency_seconds": float(np.mean(latencies))}
        rows.append(row)
    out_json, out_csv = Path(args.out_json), Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows, "protocol_assertions": "PASS",
                                    "interpretation_scope": "candidate-restricted matched evidence"}, indent=2) + "\n")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    provenance_files = list(root.glob("*.jsonl")) + [root.parent / "frozen_test_config.json"]
    provenance = {"files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in provenance_files if path.is_file()},
                  "claims": "engineering outputs only; utility requires completed real runs"}
    (out_json.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
