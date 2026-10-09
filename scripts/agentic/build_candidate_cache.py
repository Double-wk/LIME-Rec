"""Build one immutable candidate/evidence cache shared by all conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lime_rec.agentic.candidates import build_candidate_pool
from lime_rec.agentic.tools import score_hash
from lime_rec.evaluation import ExpertEvaluator
from lime_rec.protocols import prediction_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sasrec-model", required=True)
    parser.add_argument("--itemcf-model", required=True)
    parser.add_argument("--semantic-emb", required=True)
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--candidate-per-expert", type=int, default=20, choices=[20, 30, 50])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--sample-seed", type=int, default=2027)
    parser.add_argument("--out", required=True)
    parser.add_argument("--users-out")
    args = parser.parse_args()
    config_fingerprint = hashlib.sha256(json.dumps({
        "config": args.config, "sasrec": args.sasrec_model, "itemcf": args.itemcf_model,
        "semantic": args.semantic_emb, "split": args.split,
        "candidate_per_expert": args.candidate_per_expert, "max_users": args.max_users,
        "sample_seed": args.sample_seed}, sort_keys=True).encode()).hexdigest()
    out = Path(args.out)
    metadata_path = out.with_suffix(out.suffix + ".metadata.json")
    if out.exists() or metadata_path.exists():
        if out.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()).get("config_hash") == config_fingerprint:
            print(f"[cache] reuse {out}")
            if args.users_out:
                users_path = Path(args.users_out)
                users_path.parent.mkdir(parents=True, exist_ok=True)
                users = [json.loads(line)["user_id"] for line in out.read_text().splitlines() if line]
                users_path.write_text("".join(user + "\n" for user in users), encoding="utf-8")
            return
        raise RuntimeError("candidate cache exists with a different or incomplete configuration")
    evaluator = ExpertEvaluator.from_paths(args.config, args.sasrec_model, args.itemcf_model,
                                           args.semantic_emb, device=args.device, mask_history=False)
    users = list(evaluator.dataset.user_ids)
    if args.max_users and len(users) > args.max_users:
        rng = np.random.default_rng(args.sample_seed)
        users = sorted(rng.choice(users, args.max_users, replace=False).tolist())
    rows = []
    for user_id in users:
        context = prediction_context(evaluator.dataset, user_id, args.split)
        raw = evaluator.normalized_scores(user_id, context.history)
        full = {name: scores for name, scores in zip(("sasrec", "itemcf", "semantic"), raw)}
        pool = build_candidate_pool(user_id, evaluator.dataset.item_ids, full,
                                    args.candidate_per_expert)
        indices = [evaluator.sasrec.item_index[item] for item in pool.item_ids]
        restricted = np.stack([scores[indices] for scores in raw]).astype(np.float32)
        history_set = set(context.history)
        rows.append({"user_id": user_id, "split": args.split, "history": list(context.history),
                     "history_length": len(context.history), "candidate_ids": list(pool.item_ids),
                     "candidate_hash": pool.candidate_hash,
                     "history_mask": [item in history_set for item in pool.item_ids],
                     "expert_scores": restricted.tolist(),
                     "expert_score_hashes": [score_hash(row) for row in restricted]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {"dataset": evaluator.dataset.name, "split": args.split,
                "candidate_per_expert": args.candidate_per_expert,
                "sample_seed": args.sample_seed, "num_users": len(rows),
                "contains_evaluation_targets": False, "config_hash": config_fingerprint,
                "artifacts": {"sasrec": args.sasrec_model, "itemcf": args.itemcf_model,
                              "semantic": args.semantic_emb}}
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.users_out:
        users_path = Path(args.users_out)
        users_path.parent.mkdir(parents=True, exist_ok=True)
        users_path.write_text("".join(user + "\n" for user in users), encoding="utf-8")


if __name__ == "__main__":
    main()
