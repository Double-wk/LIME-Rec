"""Diagnose candidate-pool mismatch between a cached build and live recompute."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lime_rec.agentic.candidates import build_candidate_pool  # noqa: E402
from lime_rec.protocols import prediction_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--sasrec-model", required=True)
    parser.add_argument("--itemcf-model", required=True)
    parser.add_argument("--semantic-emb", required=True)
    parser.add_argument("--max-users", type=int, default=300)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    import torch
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from lime_rec.evaluation import ExpertEvaluator

    cache = Path(args.cache)
    meta = json.loads(Path(str(cache) + ".metadata.json").read_text())
    rows = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    evaluator = ExpertEvaluator.from_paths(
        args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb,
        device="cuda", mask_history=False)
    dataset = evaluator.dataset
    TOOLS = ("sasrec", "itemcf", "semantic")

    mismatches = 0
    boundary_info = []
    stats = {t: [] for t in TOOLS}
    for n, base in enumerate(rows[args.start: args.start + args.max_users]):
        uid = base["user_id"]
        report_index = args.start + n
        context = prediction_context(dataset, uid, meta["split"])
        live = evaluator.normalized_scores(uid, context.history)
        pool = build_candidate_pool(uid, dataset.item_ids, dict(zip(TOOLS, live)),
                                    meta["candidate_per_expert"])
        same = (list(pool.item_ids) == base["candidate_ids"]
                and pool.candidate_hash == base["candidate_hash"])
        if not same:
            mismatches += 1
            # per-expert boundary analysis for the first few mismatches
            if len(boundary_info) < 5:
                info = {"user": uid, "index": report_index, "cached_ids": len(base["candidate_ids"]),
                        "live_ids": len(pool.item_ids),
                        "cached_not_live": sorted(set(base["candidate_ids"]) - set(pool.item_ids))[:6],
                        "live_not_cached": sorted(set(pool.item_ids) - set(base["candidate_ids"]))[:6]}
                # score deltas on intersections
                indices = [evaluator.sasrec.item_index[i] for i in base["candidate_ids"]]
                cached = np.asarray(base["expert_scores"], dtype=np.float32)
                restricted = np.stack([v[indices] for v in live]).astype(np.float32)
                info["max_abs_delta"] = float(np.abs(cached - restricted).max())
                boundary_info.append(info)
    print(f"rows_checked={report_index + 1} pool_mismatches={mismatches}")
    for entry in boundary_info:
        print(json.dumps(entry))


if __name__ == "__main__":
    main()