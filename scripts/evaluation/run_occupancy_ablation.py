"""History-occupancy ablation for the repeat-aware gate.

Retrains the repeat-aware gate with the SAME procedure as
run_repeat_aware_gate.py (imported, not duplicated), then at test time
reports, for four penalty regimes, the full metric suite
(R/N@{5,10,20}) and the fraction of each user's Top-{5,10,20} fused
recommendations occupied by history items:

  - no_penalty_counterfactual : the penalty-TRAINED gate, penalty disabled
                                at test time only (a counterfactual: same
                                weights as `soft`, penalty term removed).
  - no_penalty_trained        : an INDEPENDENT gate trained with the penalty
                                term forced to zero throughout training.
  - soft                      : fused scores minus the per-user LEARNED penalty.
  - hard_mask                 : history items forced to -inf.

This is the evidence table for "soft != hard": under soft calibration
history items remain in the candidate set (non-zero occupancy) yet R@10
improves over the no-penalty regimes, while hard_mask drives occupancy to
exactly 0.

Occupancy is accumulated globally (history hits / user count) rather than
averaged per batch, so the final short batch is not over-weighted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lime_rec.evaluation import ExpertEvaluator
from scripts.evaluation.run_repeat_aware_gate import (
    RepeatAwareGate,
    batch_scores,
    collect_examples,
    compute_metrics,
    features,
)

OCCUPANCY_KS = (5, 10, 20)


def train_gate(evaluator, validation, args, disable_penalty: bool = False):
    """Train the repeat-aware gate identically to run_repeat_aware_gate.py.

    When ``disable_penalty`` is set, the ``- penalty * history`` term is
    dropped from the fused score during training, yielding an independently
    trained no-penalty gate (rather than a penalty-trained gate with the
    penalty switched off only at test time).
    """
    initial = [float(v) for v in args.initial_weights.split(",")]
    gate = RepeatAwareGate(14, initial, args.max_penalty).to(args.device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.l2)
    for epoch in range(1, args.epochs + 1):
        gate.train()
        for start in range(0, len(validation), args.batch_size):
            scores, lengths, targets, history = batch_scores(
                evaluator, validation[start:start + args.batch_size], args.device
            )
            weights, penalty = gate(features(scores, lengths, history))
            fused = (weights.unsqueeze(2) * scores).sum(dim=1)
            if not disable_penalty:
                fused = fused - penalty.unsqueeze(1) * history
            loss = torch.nn.functional.cross_entropy(args.score_scale * fused, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return gate


def _accumulate_occupancy(acc: dict, fused: torch.Tensor, history: torch.Tensor) -> None:
    """Add this batch's Top-k history hits and user count to global counters."""
    for k in OCCUPANCY_KS:
        topk = torch.topk(fused, k=k, dim=1).indices
        acc[k]["hits"] += float(history.gather(1, topk).float().sum().cpu())
        acc[k]["users"] += fused.shape[0]


def evaluate(evaluator, gate, no_penalty_gate, test, args):
    """Return full metrics + Top-{5,10,20} occupancy for the four regimes.

    ``gate`` is the penalty-trained gate (drives soft / hard_mask /
    no_penalty_counterfactual); ``no_penalty_gate`` is the independently
    trained no-penalty gate.
    """
    gate.eval()
    no_penalty_gate.eval()
    regimes = ("no_penalty_counterfactual", "no_penalty_trained", "soft", "hard_mask")
    scores_acc = {name: [] for name in regimes}
    occ_acc = {name: {k: {"hits": 0.0, "users": 0} for k in OCCUPANCY_KS} for name in regimes}
    # Gate outputs summarised for the soft (penalty-trained) gate.
    weights_acc, penalty_acc = [], []
    targets_all = []
    with torch.no_grad():
        for start in range(0, len(test), args.batch_size):
            scores, lengths, targets, history = batch_scores(
                evaluator, test[start:start + args.batch_size], args.device
            )
            feats = features(scores, lengths, history)
            weights, penalty = gate(feats)
            base = (weights.unsqueeze(2) * scores).sum(dim=1)

            np_weights, _ = no_penalty_gate(feats)
            np_base = (np_weights.unsqueeze(2) * scores).sum(dim=1)

            variants = {
                "no_penalty_counterfactual": base,
                "no_penalty_trained": np_base,
                "soft": base - penalty.unsqueeze(1) * history,
                "hard_mask": base.masked_fill(history, float("-inf")),
            }
            for name, fused in variants.items():
                scores_acc[name].append(fused.cpu().numpy())
                _accumulate_occupancy(occ_acc[name], fused, history)

            weights_acc.append(weights.cpu().numpy())
            penalty_acc.append(penalty.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    targets_cat = np.concatenate(targets_all)
    weights_cat = np.concatenate(weights_acc)
    penalty_cat = np.concatenate(penalty_acc)

    result = {}
    for name in regimes:
        metrics = compute_metrics(np.concatenate(scores_acc[name]), targets_cat)
        occ = {
            f"top{k}_history_occupancy": (
                occ_acc[name][k]["hits"] / (occ_acc[name][k]["users"] * k)
            )
            for k in OCCUPANCY_KS
        }
        result[name] = {**metrics, **occ}

    gate_summary = {
        "mean_weights": weights_cat.mean(axis=0).tolist(),
        "std_weights": weights_cat.std(axis=0, ddof=1).tolist(),
        "mean_penalty": float(penalty_cat.mean()),
        "std_penalty": float(penalty_cat.std(ddof=1)),
    }
    return result, gate_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sasrec-model", required=True)
    parser.add_argument("--itemcf-model", required=True)
    parser.add_argument("--semantic-emb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--score-scale", type=float, default=20.0)
    parser.add_argument("--max-penalty", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-weights", default="0.60,0.15,0.25")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    evaluator = ExpertEvaluator.from_paths(
        args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb,
        device=args.device, mask_history=False,
    )
    validation = collect_examples(evaluator, "val")
    test = collect_examples(evaluator, "test")
    print(f"[data] {evaluator.dataset.name}: val={len(validation)} test={len(test)}", flush=True)

    gate = train_gate(evaluator, validation, args, disable_penalty=False)
    no_penalty_gate = train_gate(evaluator, validation, args, disable_penalty=True)
    regimes, gate_summary = evaluate(evaluator, gate, no_penalty_gate, test, args)

    output = {
        "dataset": evaluator.dataset.name,
        "seed": args.seed,
        "protocol": {
            "gate_fit_split": "validation",
            "test_used_for_training_or_selection": False,
            "mask_history": False,
            "ranking": "full_catalog",
        },
        "hyperparameters": vars(args),
        "occupancy_ks": list(OCCUPANCY_KS),
        "regimes": regimes,
        "soft_gate_summary": gate_summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    header = f"{'regime':<28}{'R@10':>9}{'N@10':>9}{'occ@5':>9}{'occ@10':>9}{'occ@20':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in regimes.items():
        print(f"{name:<28}{m['R@10']:>9.4f}{m['N@10']:>9.4f}"
              f"{m['top5_history_occupancy']*100:>8.1f}%"
              f"{m['top10_history_occupancy']*100:>8.1f}%"
              f"{m['top20_history_occupancy']*100:>8.1f}%", flush=True)
    print(f"[gate] mean_weights={gate_summary['mean_weights']} "
          f"mean_penalty={gate_summary['mean_penalty']:.4f}+/-{gate_summary['std_penalty']:.4f}",
          flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
