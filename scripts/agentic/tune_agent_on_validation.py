"""Fit recovery and freeze an agent configuration using validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from lime_rec.agentic.recovery import CandidateRecovery
from lime_rec.evaluation import load_configured_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--recovery-out", required=True)
    parser.add_argument("--frozen-config-out", required=True)
    parser.add_argument("--agent-prompt", default="prompts/agentic_v1.txt")
    parser.add_argument("--one-shot-prompt", default="prompts/one_shot_v1.txt")
    parser.add_argument("--model", default="rule-based-mock")
    parser.add_argument("--candidate-per-expert", type=int, default=20, choices=[20, 30, 50])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dataset = load_configured_dataset(args.config)
    rows = [json.loads(line) for line in Path(args.candidate_cache).read_text().splitlines() if line]
    cache_metadata = json.loads(Path(args.candidate_cache + ".metadata.json").read_text())
    if not rows or any(row["split"] != "validation" for row in rows):
        raise RuntimeError("tuning requires a validation candidate cache")
    if cache_metadata["candidate_per_expert"] != args.candidate_per_expert:
        raise RuntimeError("candidate size must match the validation cache being tuned")
    recovery_path = Path(args.recovery_out)
    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    frozen = {"selection_split": "validation", "test_used_for_selection": False,
              "prompt_sha256": {"agentic": digest(args.agent_prompt),
                                "one_shot": digest(args.one_shot_prompt)},
              "model": args.model, "candidate_per_expert": args.candidate_per_expert,
              "temperature": args.temperature, "max_tokens": args.max_tokens,
              "max_tool_calls": args.max_tool_calls, "retry_count": args.retry_count,
              "seed": args.seed, "validation_candidate_cache_sha256": digest(args.candidate_cache),
              "recovery_checkpoint": str(recovery_path)}
    frozen_path = Path(args.frozen_config_out)
    if frozen_path.exists() or recovery_path.exists():
        if (frozen_path.exists() and recovery_path.exists()
                and json.loads(frozen_path.read_text()) == frozen):
            print(f"[freeze] reuse {frozen_path}")
            return
        raise RuntimeError("frozen config/recovery already exists with different or incomplete state")
    for row in rows:
        target = dataset.valid_by_user[row["user_id"]]
        row["target_index"] = (row["candidate_ids"].index(target)
                               if target in row["candidate_ids"] else None)
    recovery = CandidateRecovery()
    losses = recovery.fit_records(rows, split="validation", epochs=args.epochs)
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": recovery.state_dict(), "history": "candidate_only",
                "fit_split": "validation", "losses": losses}, recovery_path)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")


if __name__ == "__main__":
    main()
