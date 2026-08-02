"""Validation-based gate selection for LIME-Rec 3-expert fusion.

This script implements the standard validation-then-test protocol to prevent
test-set tuning of fusion gate weights:

  Phase 1: Grid search (w_sem, w_cf) on VALIDATION set
    - history = ds.history_by_user[u]   (train history only)
    - target  = ds.valid_by_user[u]     (held-out validation item)
  Phase 2: Apply best (w_sem, w_cf) on TEST set ONCE
    - history = ds.history_by_user[u] + [ds.valid_by_user[u]]
    - target  = ds.test_by_user[u]

Outputs:
  outputs/gate_selection_{dataset}.json
    - full validation grid (every (w_sem, w_cf) with val R@10)
    - selected (w_sem, w_cf)
  outputs/3expert_ce_{dataset}_val_selected.json
    - test eval with selected weights, same shape as eval_3expert.py output

Usage:
  python3 -m scripts.evaluation.run_gate_selection \
    --config configs/amazon_beauty.json \
    --sasrec-model outputs/models/amazon_beauty_sasrec.pt \
    --itemcf-model outputs/models/amazon_beauty_itemcf.json \
    --semantic-emb outputs/embeddings/amazon_beauty_minilm.npz \
    --sem-grid 0.05,0.10,0.15,0.20,0.25,0.30 \
    --cf-grid  0.10,0.15,0.20,0.25
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lime_rec.evaluation import ExpertEvaluator, normalize_scores


# ---------- Cached scoring (compute SASRec/ItemCF/Semantic ONCE per user) ----------


def precompute_user_scores(target_mode, evaluator, batch_size=256):
    """Return list of (user_id, target_idx, history_len, sa_n, ci_n, se_n).

    target_mode='val':  history = history_by_user;            target = valid_by_user
    target_mode='test': history = history_by_user + [valid];  target = test_by_user
    """
    ds = evaluator.dataset
    item_index = evaluator.sasrec.item_index

    # Collect valid (user, history, target) triples first.
    pending = []  # (user_id, history, target_idx)
    for u in ds.user_ids:
        if target_mode == "val":
            history = ds.history_by_user.get(u, [])
            target_item = ds.valid_by_user.get(u, "")
        else:  # test
            history = ds.history_by_user.get(u, []) + [ds.valid_by_user.get(u, "")]
            history = [h for h in history if h]
            target_item = ds.test_by_user.get(u, "")
        if not target_item or not history:
            continue
        target = item_index.get(target_item)
        if target is None:
            continue
        pending.append((u, history, target))

    out = []
    for start in tqdm(range(0, len(pending), batch_size),
                      desc=f"score-{target_mode}", unit="batch"):
        chunk = pending[start:start + batch_size]
        user_ids = [c[0] for c in chunk]
        histories = [c[1] for c in chunk]
        sa, ci, se = evaluator.score_batch(user_ids, histories)
        for row, (u, history, target) in enumerate(chunk):
            sa_n = normalize_scores(sa[row])
            ci_n = normalize_scores(ci[row])
            se_n = normalize_scores(se[row])
            hist_idx = np.array(
                sorted({item_index[h] for h in history if h in item_index}),
                dtype=np.int64,
            )
            k = min(5, sa_n.size)
            contamination_rates = np.asarray([
                float(np.isin(np.argpartition(scores, -k)[-k:], hist_idx).sum()) / k
                for scores in (sa_n, ci_n, se_n)
            ], dtype=np.float32)
            out.append((u, target, len(history), sa_n, ci_n, se_n, hist_idx, contamination_rates))
    return out


def history_contamination(
    expert_scores, expert_weights, history_indices, topk=5
):
    """Weighted fraction of each expert's Top-K occupied by history items."""
    if history_indices.size == 0:
        return 0.0
    k = min(topk, expert_scores[0].size)
    contamination = 0.0
    for weight, scores in zip(expert_weights, expert_scores):
        top_indices = np.argpartition(scores, -k)[-k:]
        overlap = np.isin(top_indices, history_indices).sum()
        contamination += weight * float(overlap) / k
    return contamination


def evaluate_gate(
    scored_users,
    w_seq,
    w_cf,
    w_sem,
    history_penalty=0.0,
    penalty_mode="fixed",
    contamination_topk=5,
    max_effective_penalty=math.inf,
    ks=(5, 10, 20),
):
    """Evaluate one fused gate over all (cached) users.
    Returns dict with mean R@k and N@k.
    """
    metrics = {f"R@{k}": [] for k in ks}
    metrics.update({f"N@{k}": [] for k in ks})

    for u, target, _hl, sa_n, ci_n, se_n, hist_idx, contamination_rates in scored_users:
        f = w_seq * sa_n + w_cf * ci_n + w_sem * se_n
        if history_penalty and hist_idx.size:
            effective_penalty = history_penalty
            if penalty_mode == "adaptive" and math.isfinite(history_penalty):
                if contamination_topk == 5:
                    contamination = float(np.dot(
                        (w_seq, w_cf, w_sem), contamination_rates
                    ))
                else:
                    contamination = history_contamination(
                        (sa_n, ci_n, se_n),
                        (w_seq, w_cf, w_sem),
                        hist_idx,
                        contamination_topk,
                    )
                effective_penalty *= contamination
            effective_penalty = min(effective_penalty, max_effective_penalty)
            f = f.copy()
            f[hist_idx] -= effective_penalty
        # rank of target item: count items with strictly higher score
        rank = int((f > f[target]).sum())
        for k in ks:
            hit = 1.0 if rank < k else 0.0
            ndcg = 1.0 / math.log2(rank + 2) if rank < k else 0.0
            metrics[f"R@{k}"].append(hit)
            metrics[f"N@{k}"].append(ndcg)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def evaluate_single_expert(scored_users, expert_idx, ks=(5, 10, 20)):
    """Evaluate a single expert (index 0/1/2 = seq/cf/sem) without fusion."""
    metrics = {f"R@{k}": [] for k in ks}
    metrics.update({f"N@{k}": [] for k in ks})
    for _u, target, _hl, sa_n, ci_n, se_n, _history_indices, _contamination in scored_users:
        s = (sa_n, ci_n, se_n)[expert_idx]
        rank = int((s > s[target]).sum())
        for k in ks:
            hit = 1.0 if rank < k else 0.0
            ndcg = 1.0 / math.log2(rank + 2) if rank < k else 0.0
            metrics[f"R@{k}"].append(hit)
            metrics[f"N@{k}"].append(ndcg)
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ---------- Main protocol ----------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sasrec-model", required=True)
    p.add_argument("--itemcf-model", required=True)
    p.add_argument("--semantic-emb", required=True)
    p.add_argument("--sem-grid", default="0.05,0.10,0.15,0.20,0.25,0.30",
                   help="Validation grid for semantic weight")
    p.add_argument("--cf-grid", default="0.10,0.15,0.20,0.25",
                   help="Validation grid for co-occurrence weight")
    p.add_argument("--history-penalty-grid", default="0,0.02,0.05,0.10,0.15,0.20,0.30",
                   help="Validation grid for soft penalties subtracted from fused scores of history items")
    p.add_argument("--history-penalty-mode", default="fixed",
                   choices=["fixed", "adaptive"],
                   help="Fixed penalty or per-user penalty scaled by expert Top-K history contamination")
    p.add_argument("--contamination-topk", type=int, default=5,
                   help="Expert Top-K used to estimate per-user history contamination")
    p.add_argument("--max-effective-penalty", type=float, default=math.inf,
                   help="Upper bound on the per-user effective penalty; default is unbounded")
    p.add_argument("--select-metric", default="R@10",
                   choices=["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"],
                   help="Validation metric for gate selection")
    p.add_argument("--out-gate", default=None,
                   help="Path to gate_selection_{dataset}.json; default outputs/gate_selection_{name}.json")
    p.add_argument("--out-test", default=None,
                   help="Path to test result with val-selected weights; default outputs/3expert_ce_{name}_val_selected.json")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="inference device")
    p.add_argument("--no-mask", action="store_true",
                   help="do not exclude interacted history items (no-mask protocol)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="users per inference batch during score precompute")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    name = cfg["name"]
    out_gate = args.out_gate or f"outputs/gate_selection_{name}.json"
    out_test = args.out_test or f"outputs/3expert_ce_{name}_val_selected.json"

    evaluator = ExpertEvaluator.from_paths(
        args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb,
        device=args.device, mask_history=not args.no_mask,
    )
    ds = evaluator.dataset
    print(f"[data] {name}: users={ds.num_users} items={ds.num_items}", flush=True)

    sem_grid = [float(x) for x in args.sem_grid.split(",")]
    cf_grid = [float(x) for x in args.cf_grid.split(",")]
    penalty_grid = [float(x) for x in args.history_penalty_grid.split(",")]
    if not penalty_grid or any(value < 0 or not math.isfinite(value) for value in penalty_grid):
        p.error("--history-penalty-grid must contain finite non-negative values")
    if args.contamination_topk < 1:
        p.error("--contamination-topk must be positive")
    if args.max_effective_penalty <= 0 or math.isnan(args.max_effective_penalty):
        p.error("--max-effective-penalty must be positive")

    # ===== PHASE 1: validation grid =====
    print(f"\n[phase-1] computing validation scores for {len(ds.user_ids)} users", flush=True)
    val_users = precompute_user_scores("val", evaluator, batch_size=args.batch_size)
    print(f"[phase-1] valid users with valid target: {len(val_users)}", flush=True)

    # Per-expert val metrics (for sanity check + JSON record)
    val_single = {
        "SASRec_only": evaluate_single_expert(val_users, 0),
        "ItemCF_only": evaluate_single_expert(val_users, 1),
        "Semantic_only": evaluate_single_expert(val_users, 2),
    }

    grid_results = []
    print("\n[phase-1] gate grid search on validation:")
    print(f"  {'w_seq':>8}{'w_cf':>8}{'w_sem':>8}{'penalty':>10}    {args.select_metric:>8}")
    for w_sem in sem_grid:
        for w_cf in cf_grid:
            w_seq = 1.0 - w_sem - w_cf
            if w_seq <= 0:
                continue
            for penalty in penalty_grid:
                m = evaluate_gate(
                    val_users, w_seq, w_cf, w_sem,
                    history_penalty=penalty,
                    penalty_mode=args.history_penalty_mode,
                    contamination_topk=args.contamination_topk,
                    max_effective_penalty=args.max_effective_penalty,
                )
                entry = {
                    "w_seq": w_seq, "w_cf": w_cf, "w_sem": w_sem,
                    "history_penalty": penalty,
                    "history_penalty_mode": args.history_penalty_mode,
                    "val_metrics": m,
                }
                grid_results.append(entry)
                sel_val = m[args.select_metric]
                print(f"  {w_seq:>8.3f}{w_cf:>8.3f}{w_sem:>8.3f}{penalty:>10.3f}    {sel_val:>8.4f}", flush=True)

    # Select the best gate and penalty by validation metric; break ties toward no penalty.
    best = max(grid_results, key=lambda e: (e["val_metrics"][args.select_metric], -e["history_penalty"]))
    print(f"\n[selected] w_seq={best['w_seq']:.3f}  w_cf={best['w_cf']:.3f}  w_sem={best['w_sem']:.3f}"
          f"  penalty={best['history_penalty']:.3f}"
          f"  val {args.select_metric}={best['val_metrics'][args.select_metric]:.4f}", flush=True)

    # Write gate_selection json
    Path(out_gate).parent.mkdir(parents=True, exist_ok=True)
    Path(out_gate).write_text(json.dumps({
        "dataset": name,
        "sasrec_backbone": str(args.sasrec_model),
        "semantic_emb": str(args.semantic_emb),
        "select_metric": args.select_metric,
        "sem_grid": sem_grid,
        "cf_grid": cf_grid,
        "history_penalty_grid": penalty_grid,
        "history_penalty_mode": args.history_penalty_mode,
        "contamination_topk": args.contamination_topk,
        "max_effective_penalty": args.max_effective_penalty,
        "protocol": {"mask_history": not args.no_mask, "ranking": "full_catalog"},
        "single_expert_val": val_single,
        "grid": grid_results,
        "selected": {
            "w_seq": best["w_seq"], "w_cf": best["w_cf"], "w_sem": best["w_sem"],
            "history_penalty": best["history_penalty"],
            "history_penalty_mode": args.history_penalty_mode,
            "val_metrics": best["val_metrics"],
        },
    }, indent=2))
    print(f"[saved] {out_gate}", flush=True)

    # ===== PHASE 2: test eval with selected weights (ONCE) =====
    print(f"\n[phase-2] computing test scores for {len(ds.user_ids)} users", flush=True)
    test_users = precompute_user_scores("test", evaluator, batch_size=args.batch_size)
    print(f"[phase-2] valid users with test target: {len(test_users)}", flush=True)

    # Test eval: single experts + selected fused gate
    test_single = {
        "SASRec": evaluate_single_expert(test_users, 0),
        "ItemCF": evaluate_single_expert(test_users, 1),
        "Semantic": evaluate_single_expert(test_users, 2),
    }
    test_fused_no_penalty = evaluate_gate(
        test_users, best["w_seq"], best["w_cf"], best["w_sem"], history_penalty=0.0
    )
    test_fused_soft = evaluate_gate(
        test_users, best["w_seq"], best["w_cf"], best["w_sem"],
        history_penalty=best["history_penalty"],
        penalty_mode=args.history_penalty_mode,
        contamination_topk=args.contamination_topk,
        max_effective_penalty=args.max_effective_penalty,
    )
    test_fused_fixed_same_base = evaluate_gate(
        test_users, best["w_seq"], best["w_cf"], best["w_sem"],
        history_penalty=best["history_penalty"], penalty_mode="fixed",
        contamination_topk=args.contamination_topk,
        max_effective_penalty=args.max_effective_penalty,
    )
    test_fused_hard = evaluate_gate(
        test_users, best["w_seq"], best["w_cf"], best["w_sem"], history_penalty=math.inf
    )

    summary = {
        "SASRec": {"all": test_single["SASRec"]},
        "ItemCF": {"all": test_single["ItemCF"]},
        "Semantic": {"all": test_single["Semantic"]},
        "3expert_no_penalty": {
            "all": test_fused_no_penalty,
            "weights": {"w_seq": best["w_seq"], "w_cf": best["w_cf"], "w_sem": best["w_sem"]},
            "history_penalty": 0.0,
        },
        f"3expert_{args.history_penalty_mode}_penalty": {
            "all": test_fused_soft,
            "weights": {"w_seq": best["w_seq"], "w_cf": best["w_cf"], "w_sem": best["w_sem"]},
            "history_penalty": best["history_penalty"],
            "history_penalty_mode": args.history_penalty_mode,
        },
        "3expert_hard_mask": {
            "all": test_fused_hard,
            "weights": {"w_seq": best["w_seq"], "w_cf": best["w_cf"], "w_sem": best["w_sem"]},
            "history_penalty": "infinity",
        },
    }
    if args.history_penalty_mode == "adaptive":
        summary["3expert_fixed_same_base"] = {
            "all": test_fused_fixed_same_base,
            "weights": {"w_seq": best["w_seq"], "w_cf": best["w_cf"], "w_sem": best["w_sem"]},
            "history_penalty": best["history_penalty"],
            "history_penalty_mode": "fixed",
        }

    print(f"\n{'Method':<30}{'R@5':>9}{'N@5':>9}{'R@10':>9}{'N@10':>9}{'R@20':>9}{'N@20':>9}")
    print("-" * 84)
    for k, v in summary.items():
        s = v["all"]
        print(f"{k:<30}{s['R@5']:>9.4f}{s['N@5']:>9.4f}"
              f"{s['R@10']:>9.4f}{s['N@10']:>9.4f}"
              f"{s['R@20']:>9.4f}{s['N@20']:>9.4f}")

    Path(out_test).parent.mkdir(parents=True, exist_ok=True)
    Path(out_test).write_text(json.dumps({
        "dataset": name,
        "sasrec_backbone": str(args.sasrec_model),
        "semantic_emb": str(args.semantic_emb),
        "gate_selection": str(out_gate),
        "selected_weights": {
            "w_seq": best["w_seq"], "w_cf": best["w_cf"], "w_sem": best["w_sem"],
            "history_penalty": best["history_penalty"],
            "history_penalty_mode": args.history_penalty_mode,
        },
        "select_metric": args.select_metric,
        "protocol": {"mask_history": not args.no_mask, "ranking": "full_catalog"},
        "contamination_topk": args.contamination_topk,
        "max_effective_penalty": args.max_effective_penalty,
        "summary": summary,
    }, indent=2))
    print(f"[saved] {out_test}", flush=True)


if __name__ == "__main__":
    main()
