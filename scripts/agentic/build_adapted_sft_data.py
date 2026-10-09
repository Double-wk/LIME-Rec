"""Build SFT data for the task-adapted controller condition.

Two supervision sources share one byte-identical message format (system =
one_shot prompt, user = the all-tools-observed payload ending with "You must now
return final ranking.", assistant = the finish action JSON):

- ``--supervision witness`` (default): imitation of the validation-fitted
  deterministic recovery path (CandidateRecovery) applied to each validation
  user's shared candidate pool. This is the distillation target: the controller
  learns to reproduce the witness ranking.
- ``--supervision label``: task supervision. The top-1 of the completion is the
  user's true held-out validation item, so the controller learns to identify the
  correct item from the same evidence rather than to copy the witness. Only
  users whose validation target falls inside their candidate pool can be
  supervised this way (~19.5% of validation users); the remaining nine slots
  keep the witness order so the demonstration stays a valid pool ranking.

Only the validation split is used; test users/labels are never read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lime_rec.agentic.recovery import CandidateRecovery
from lime_rec.evaluation import load_configured_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--recovery-checkpoint", required=True)
    parser.add_argument("--prompt", default="prompts/one_shot_v1.txt")
    parser.add_argument("--supervision", choices=["witness", "label"], default="witness",
                        help="witness = imitate the deterministic ranking; "
                             "label = put the true held-out validation item first.")
    parser.add_argument("--config", help="Dataset config; required for --supervision label.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.candidate_cache).read_text().splitlines() if line]
    prompt = Path(args.prompt).read_text()
    checkpoint = torch.load(args.recovery_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["fit_split"] != "validation":
        raise RuntimeError("recovery teacher must be validation-fitted")
    valid_by_user = None
    if args.supervision == "label":
        if not args.config:
            raise RuntimeError("--supervision label requires --config to resolve validation targets")
        valid_by_user = load_configured_dataset(args.config).valid_by_user
    recovery = CandidateRecovery(device="cpu")
    recovery.gate.load_state_dict(checkpoint["state_dict"])
    recovery.fitted_split = checkpoint["fit_split"]

    out_rows = []
    skipped_no_target = 0
    for row in rows:
        candidates = list(row["candidate_ids"])
        scores = np.asarray(row["expert_scores"], dtype=np.float32)
        mask = np.asarray(row["history_mask"], dtype=bool)
        fused = recovery.score(scores[None], mask[None], np.asarray([row["history_length"]]))[0]
        order = sorted(range(len(candidates)), key=lambda i: (-float(fused[i]), candidates[i]))
        # Top-10 completion: matches the recovery condition's output width and stays
        # well within the frozen max_tokens=512 test-time budget.
        if args.supervision == "witness":
            teacher = [candidates[i] for i in order[:10]]
        else:
            target = valid_by_user[row["user_id"]]
            if target not in candidates:
                # No supervision is available for this user: the label is outside
                # the pool, so no admissible pool ranking can contain it.
                skipped_no_target += 1
                continue
            rest = [candidates[i] for i in order if candidates[i] != target]
            teacher = [target] + rest[:9]

        # Rebuild the exact tool observations (same float32 values + rank rules
        # as lime_rec.agentic.tools.ExpertTool.query).
        observations = {}
        for name, expert_idx in (("sasrec", 0), ("itemcf", 1), ("semantic", 2)):
            values = scores[expert_idx]
            rank_order = sorted(range(len(candidates)),
                                key=lambda i: (-float(values[i]), str(candidates[i])))
            ranks = {idx: r for r, idx in enumerate(rank_order, start=1)}
            observations[name] = {
                "tool": name,
                "score_hash": hashlib.sha256(
                    np.asarray(values, dtype="<f4").tobytes()).hexdigest(),
                "items": [
                    {"item_id": str(item), "score": float(values[i]),
                     "rank_within_candidate_pool": ranks[i]}
                    for i, item in enumerate(candidates)
                ],
            }

        payload = {
            "history_length": row["history_length"],
            "candidate_ids": candidates,
            "is_history_item": [bool(x) for x in row["history_mask"]],
            "available_tools": ["sasrec", "itemcf", "semantic"],
            "called_tools": ["sasrec", "itemcf", "semantic"],
            "remaining_tool_budget": 0,
            "observations": observations,
        }
        user_msg = "You must now return final ranking.\n" + json.dumps(payload)
        assistant_msg = json.dumps({"action": "finish", "ranking": teacher})
        out_rows.append({"system": prompt, "prompt": user_msg, "completion": assistant_msg,
                         "user_id": row["user_id"]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for r in out_rows:
            handle.write(json.dumps(r) + "\n")
    print(f"[sft] wrote {len(out_rows)} examples -> {out}")
    if args.supervision == "label":
        print(f"[sft] supervision=label; skipped {skipped_no_target} users whose "
              f"validation target lies outside their candidate pool")


if __name__ == "__main__":
    main()
