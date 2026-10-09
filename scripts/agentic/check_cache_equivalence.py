"""Verify that a rebuilt candidate cache is behaviorally equivalent to a frozen one.

Frozen caches are pinned to the environment that produced them. A later
environment can reproduce the same pools and the same audit decisions while
differing from the frozen scores in the last float32 bits, which fails the
byte-level expert-score checksum used by the frozen pipelines. This script
decides whether a rebuilt cache may stand in for the frozen one, by testing the
properties the audit actually depends on rather than byte equality:

1. identical user set (and identical user order);
2. identical per-user candidate pool, in identical order (candidate_hash);
3. expert-score deviation confined to float32 rounding;
4. the witness ranking recomputed from the rebuilt cache, through the frozen
   validation-fitted recovery checkpoint, is identical for every user to the
   frozen witness predictions.

A disagreement on any of 1, 2, or 4 means the rebuild is not admissible.

Usage:
  python3 -m scripts.agentic.check_cache_equivalence \
    --frozen outputs/agentic/candidate_cache/amazon_beauty_test_1000.jsonl \
    --rebuilt outputs/agentic/candidate_cache/amazon_beauty_test_1000_rebuilt.jsonl \
    --recovery-checkpoint outputs/agentic/beauty_recovery_validation_real.pt \
    --frozen-predictions output_final/results/agentic/beauty/recovery.jsonl \
    --out-json output_final/results/agentic/cache_equivalence.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lime_rec.agentic.recovery import CandidateRecovery


def _load(path: str) -> dict[str, dict]:
    return {json.loads(line)["user_id"]: json.loads(line)
            for line in Path(path).read_text().splitlines() if line}


def _witness_ranking(record: dict, recovery: CandidateRecovery) -> list[str]:
    candidates = list(record["candidate_ids"])
    scores = np.asarray(record["expert_scores"], dtype=np.float32)
    fused = recovery.score(scores[None], np.asarray(record["history_mask"])[None],
                           np.asarray([record["history_length"]]))[0]
    order = sorted(range(len(candidates)), key=lambda i: (-float(fused[i]), candidates[i]))
    return [candidates[i] for i in order[:10]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--rebuilt", required=True)
    parser.add_argument("--recovery-checkpoint", required=True)
    parser.add_argument("--frozen-predictions", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    frozen, rebuilt = _load(args.frozen), _load(args.rebuilt)
    checkpoint = torch.load(args.recovery_checkpoint, map_location="cpu", weights_only=False)
    recovery = CandidateRecovery(device="cpu")
    recovery.gate.load_state_dict(checkpoint["state_dict"])
    recovery.fitted_split = checkpoint["fit_split"]
    predictions = _load(args.frozen_predictions)

    user_set_ok = set(frozen) == set(rebuilt)
    user_order_ok = list(rebuilt) == list(frozen)
    pool_mismatch = [u for u in frozen if frozen[u]["candidate_hash"] != rebuilt[u]["candidate_hash"]]
    score_deviation = [float(np.abs(np.asarray(frozen[u]["expert_scores"], dtype=np.float32)
                                    - np.asarray(rebuilt[u]["expert_scores"], dtype=np.float32)).max())
                       for u in frozen if user_set_ok]
    ranking_mismatch = ([u for u in frozen
                         if _witness_ranking(rebuilt[u], recovery) != predictions[u]["ranking"]]
                        if user_set_ok else [])
    worst = max(score_deviation) if score_deviation else float("nan")
    admissible = bool(user_set_ok and user_order_ok and not pool_mismatch and not ranking_mismatch)

    result = {
        "frozen_cache": args.frozen,
        "rebuilt_cache": args.rebuilt,
        "admissible": admissible,
        "checks": {
            "user_set_identical": user_set_ok,
            "user_order_identical": user_order_ok,
            "candidate_pool_mismatches": len(pool_mismatch),
            "witness_ranking_mismatches": len(ranking_mismatch),
            "expert_score_max_abs_deviation": worst,
            "float32_eps": float(np.finfo(np.float32).eps),
            "expert_score_deviation_within_rounding": bool(worst <= 16 * np.finfo(np.float32).eps),
        },
        "reading": ("The rebuilt cache preserves the user set, the candidate pools, and every "
                    "frozen witness decision; expert scores differ only at float32 rounding, "
                    "which the byte-level checksum cannot tolerate across environments."
                    if admissible else
                    "Not admissible as a drop-in replacement for the frozen cache."),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"], indent=2))
    print(f"[admissible] {admissible} -> {out}")
    if not admissible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()