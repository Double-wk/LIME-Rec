"""Dump per-user metrics for RQ3 complementarity analysis.

For each test user, emit one JSON line with:
  - user_id, history_length, target_idx
  - per-expert {target_rank, topK item indices in rank order}
  - fused gate {weights, target_rank, topK item indices}

Downstream `analyze_complementarity.py` consumes this to produce:
  - Jaccard@K between expert top-K sets
  - tie-corrected Spearman rank correlation over the full candidate catalog
  - Lift attribution: among users where fusion top-K hits target but SASRec
    does not, breakdown by which expert top-K contained the target.

Test-set scoring mirrors `eval_3expert.py` / `run_gate_selection.py`:
  history = ds.history_by_user[u] + [ds.valid_by_user[u]]
  target  = ds.test_by_user[u]

Weights for fusion should be the validation-selected (w_seq, w_cf, w_sem)
from `outputs/gate_selection_{dataset}.json`; if not supplied we default to
(0.55, 0.15, 0.30) which is approximately what B/T/S val-select to.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm

from lime_rec.evaluation import (
    ExpertEvaluator,
    normalize_scores,
    rank_of,
    top_k_indices,
)


def spearman_full(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Spearman ρ between two score vectors over a candidate mask.

    `mask` is a boolean array of items to include (e.g. exclude masked/history items).
    Both `a` and `b` are score vectors of the same length. We rank within `mask` only.
    """
    a_sub = a[mask]
    b_sub = b[mask]
    if a_sub.size < 2:
        return 0.0
    # ItemCF commonly assigns the same zero score to most candidates. Average
    # ranks are required for those ties; argsort().argsort() assigns arbitrary
    # distinct ranks and can create spurious correlations.
    ra = rankdata(a_sub, method="average")
    rb = rankdata(b_sub, method="average")
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0.0:
        return 0.0
    return float((ra * rb).sum() / denom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sasrec-model", required=True)
    p.add_argument("--itemcf-model", required=True)
    p.add_argument("--semantic-emb", required=True)
    p.add_argument("--gate-selection", default=None,
                   help="Path to gate_selection_{dataset}.json; if provided, "
                        "fusion weights are loaded from it. Else --w-seq/cf/sem are used.")
    p.add_argument("--w-seq", type=float, default=0.55)
    p.add_argument("--w-cf", type=float, default=0.15)
    p.add_argument("--w-sem", type=float, default=0.30)
    p.add_argument("--topk", type=int, default=20,
                   help="Number of top items to record per expert and fusion")
    p.add_argument("--max-users", type=int, default=None,
                   help="Deterministically sample at most this many eligible users")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed used with --max-users (default: 42)")
    p.add_argument("--out", default=None,
                   help="Output JSONL path; default outputs/per_user_metrics_{dataset}.jsonl")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    name = cfg["name"]
    out_path = args.out or f"outputs/per_user_metrics_{name}.jsonl"

    if args.gate_selection and Path(args.gate_selection).exists():
        gs = json.loads(Path(args.gate_selection).read_text(encoding="utf-8"))
        w_seq = gs["selected"]["w_seq"]
        w_cf = gs["selected"]["w_cf"]
        w_sem = gs["selected"]["w_sem"]
        print(f"[weights] loaded from {args.gate_selection}: "
              f"w_seq={w_seq:.3f} w_cf={w_cf:.3f} w_sem={w_sem:.3f}", flush=True)
    else:
        w_seq, w_cf, w_sem = args.w_seq, args.w_cf, args.w_sem
        print(f"[weights] from args: "
              f"w_seq={w_seq:.3f} w_cf={w_cf:.3f} w_sem={w_sem:.3f}", flush=True)

    evaluator = ExpertEvaluator.from_paths(
        args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb
    )
    ds = evaluator.dataset
    print(f"[data] {name}: users={ds.num_users} items={ds.num_items}", flush=True)

    item_index = evaluator.sasrec.item_index

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    K = args.topk
    user_ids = [u for u in ds.user_ids if ds.test_by_user.get(u, "")]
    if args.max_users is not None:
        if args.max_users <= 0:
            p.error("--max-users must be positive")
        if len(user_ids) > args.max_users:
            rng = np.random.default_rng(args.seed)
            selected = rng.choice(len(user_ids), size=args.max_users, replace=False)
            user_ids = [user_ids[i] for i in sorted(selected.tolist())]
        print(f"[sample] users={len(user_ids)} seed={args.seed}", flush=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for u in tqdm(user_ids, desc="dump-per-user", unit="user"):
            target_item = ds.test_by_user.get(u, "")
            if not target_item:
                continue
            target_idx = item_index.get(target_item)
            if target_idx is None:
                continue
            history = ds.history_by_user.get(u, []) + [ds.valid_by_user.get(u, "")]
            history = [h for h in history if h]
            if not history:
                continue

            sa, ci, se = evaluator.score(u, history)
            sa_n, ci_n, se_n = map(normalize_scores, (sa, ci, se))
            fused = w_seq * sa_n + w_cf * ci_n + w_sem * se_n

            # Full-vector Spearman ρ over non-masked items (proper measure
            # of whether experts globally agree, not biased by tiny top-K overlap).
            valid_mask = sa > -1e8  # history items were set to -1e9
            sp_sa_ci = spearman_full(sa_n, ci_n, valid_mask)
            sp_sa_se = spearman_full(sa_n, se_n, valid_mask)
            sp_ci_se = spearman_full(ci_n, se_n, valid_mask)

            entry = {
                "user_id": u,
                "history_length": len(history),
                "target_idx": int(target_idx),
                "experts": {
                    "SASRec": {
                        "target_rank": rank_of(sa_n, target_idx),
                        "topK": top_k_indices(sa_n, K).tolist(),
                    },
                    "ItemCF": {
                        "target_rank": rank_of(ci_n, target_idx),
                        "topK": top_k_indices(ci_n, K).tolist(),
                    },
                    "Semantic": {
                        "target_rank": rank_of(se_n, target_idx),
                        "topK": top_k_indices(se_n, K).tolist(),
                    },
                },
                "fusion": {
                    "3expert_val_selected": {
                        "weights": [float(w_seq), float(w_cf), float(w_sem)],
                        "target_rank": rank_of(fused, target_idx),
                        "topK": top_k_indices(fused, K).tolist(),
                    },
                },
                "spearman_full": {
                    "SASRec,ItemCF": sp_sa_ci,
                    "SASRec,Semantic": sp_sa_se,
                    "ItemCF,Semantic": sp_ci_se,
                },
            }
            fout.write(json.dumps(entry) + "\n")
            n_written += 1

    print(f"[saved] {out_path} ({n_written} users, K={K})", flush=True)


if __name__ == "__main__":
    main()
