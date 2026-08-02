"""Control experiments for the repeat-aware gate.

Reuses the exact training/eval procedure of run_repeat_aware_gate.py
(imported, not duplicated) and toggles ONE thing at a time so each control
is apples-to-apples with the main result:

  --control shuffle_embeddings
      Row-permute the semantic embedding matrix with a fixed seed before
      fitting the gate. The embedding SET (dimensionality, norm distribution,
      parameter count, cosine geometry) is unchanged; only the item->text
      correspondence is destroyed. If fusion still wins, the gain is coming
      from extra capacity rather than real text semantics. Consistently
      rewrites the user-side matrix, the catalog-normalized matrix, and the
      catalog tensor so both sides of the cosine use the same permutation.

  --control fixed_penalty
      Replace the per-user penalty head with a SINGLE global learnable scalar
      (still bounded by max_penalty * sigmoid). The three expert weights are
      still learned per user. This is the "one global penalty" baseline: if the
      per-user penalty does not beat this, the penalty is not meaningfully
      user-adaptive.

Both controls keep mask_history=False, full-catalog ranking, and gate fitting
on validation only, identical to the main method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lime_rec.evaluation import ExpertEvaluator
from scripts.evaluation.run_repeat_aware_gate import (
    RepeatAwareGate,
    batch_scores,
    collect_examples,
    compute_metrics,
    features,
)


def shuffle_semantic_embeddings(evaluator: ExpertEvaluator, seed: int) -> None:
    """Row-permute the semantic embeddings in place, consistently on both sides."""
    semantic = evaluator.semantic
    num_items = semantic.item_embeddings.shape[0]
    perm = np.random.RandomState(seed).permutation(num_items)
    permuted = semantic.item_embeddings[perm].astype(np.float32)
    norms = np.linalg.norm(permuted, axis=1, keepdims=True) + 1e-8
    semantic.item_embeddings = permuted
    semantic.item_embeddings_norm = permuted / norms
    evaluator.semantic_catalog_tensor = torch.as_tensor(
        semantic.item_embeddings_norm, dtype=torch.float32, device=evaluator.device
    )


class FixedPenaltyGate(RepeatAwareGate):
    """Per-user expert weights, but a single global (non-user) penalty scalar."""

    def __init__(self, feature_dim, initial_weights, max_penalty):
        super().__init__(feature_dim, initial_weights, max_penalty)
        # Drop the penalty row from the linear head; use one shared logit instead.
        self.penalty_logit = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor):
        output = self.linear(x)
        weights = torch.softmax(output[:, :3], dim=1)
        penalty = self.max_penalty * torch.sigmoid(self.penalty_logit)
        penalty = penalty.expand(x.shape[0])
        return weights, penalty


def build_gate(control: str, initial, args):
    if control == "fixed_penalty":
        return FixedPenaltyGate(14, initial, args.max_penalty).to(args.device)
    return RepeatAwareGate(14, initial, args.max_penalty).to(args.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sasrec-model", required=True)
    parser.add_argument("--itemcf-model", required=True)
    parser.add_argument("--semantic-emb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--control", required=True,
                        choices=["shuffle_embeddings", "fixed_penalty"])
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
    initial = [float(v) for v in args.initial_weights.split(",")]
    if len(initial) != 3 or any(v <= 0 for v in initial):
        parser.error("initial-weights must contain three positive values")

    torch.manual_seed(args.seed)
    evaluator = ExpertEvaluator.from_paths(
        args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb,
        device=args.device, mask_history=False,
    )
    if args.control == "shuffle_embeddings":
        shuffle_semantic_embeddings(evaluator, args.seed)
        print(f"[control] semantic embeddings row-permuted with seed {args.seed}", flush=True)

    validation = collect_examples(evaluator, "val")
    test = collect_examples(evaluator, "test")
    print(f"[data] {evaluator.dataset.name}: val={len(validation)} test={len(test)}", flush=True)

    gate = build_gate(args.control, initial, args)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.l2)
    for epoch in range(1, args.epochs + 1):
        gate.train()
        losses = []
        for start in tqdm(range(0, len(validation), args.batch_size),
                          desc=f"train-{epoch}", unit="batch"):
            scores, lengths, targets, history = batch_scores(
                evaluator, validation[start:start + args.batch_size], args.device
            )
            weights, penalty = gate(features(scores, lengths, history))
            fused = (weights.unsqueeze(2) * scores).sum(dim=1) - penalty.unsqueeze(1) * history
            loss = torch.nn.functional.cross_entropy(args.score_scale * fused, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[train] epoch={epoch} cross_entropy={np.mean(losses):.5f}", flush=True)

    gate.eval()
    all_scores, all_targets, all_weights, all_penalties = [], [], [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(test), args.batch_size), desc="test", unit="batch"):
            scores, lengths, targets, history = batch_scores(
                evaluator, test[start:start + args.batch_size], args.device
            )
            weights, penalty = gate(features(scores, lengths, history))
            fused = (weights.unsqueeze(2) * scores).sum(dim=1) - penalty.unsqueeze(1) * history
            all_scores.append(fused.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
            all_penalties.append(penalty.cpu().numpy())

    penalties = np.concatenate(all_penalties)
    output = {
        "dataset": evaluator.dataset.name,
        "seed": args.seed,
        "control": args.control,
        "protocol": {"gate_fit_split": "validation", "test_used_for_training_or_selection": False,
                     "mask_history": False, "ranking": "full_catalog"},
        "method": f"repeat-aware expert gate [control={args.control}]",
        "hyperparameters": vars(args),
        "test_metrics": compute_metrics(np.concatenate(all_scores), np.concatenate(all_targets)),
        "mean_test_weights": np.concatenate(all_weights).mean(axis=0).tolist(),
        "mean_test_penalty": float(penalties.mean()),
        "std_test_penalty": float(penalties.std()),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[test] {json.dumps(output['test_metrics'], sort_keys=True)}", flush=True)
    print(f"[test] mean_weights={output['mean_test_weights']} "
          f"penalty={output['mean_test_penalty']:.4f}+/-{output['std_test_penalty']:.4f}", flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
