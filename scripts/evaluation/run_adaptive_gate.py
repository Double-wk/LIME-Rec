"""Train a lightweight per-user fusion gate on validation, then test once.

The three recommender experts are frozen.  The gate sees only features available
at inference time: history length plus per-expert score-distribution statistics.
It is fitted against validation targets and is never fitted or selected on test.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from lime_rec.evaluation import ExpertEvaluator, normalize_score_rows


def collect_examples(evaluator: ExpertEvaluator, split: str) -> list[tuple[str, list[str], int]]:
    dataset = evaluator.dataset
    item_index = evaluator.sasrec.item_index
    examples: list[tuple[str, list[str], int]] = []
    for user_id in dataset.user_ids:
        if split == "val":
            history = list(dataset.history_by_user.get(user_id, []))
            target_id = dataset.valid_by_user.get(user_id, "")
        else:
            history = list(dataset.history_by_user.get(user_id, []))
            valid_id = dataset.valid_by_user.get(user_id, "")
            if valid_id:
                history.append(valid_id)
            target_id = dataset.test_by_user.get(user_id, "")
        target = item_index.get(target_id)
        if history and target is not None:
            examples.append((user_id, history, target))
    return examples


def batch_scores(
    evaluator: ExpertEvaluator, batch: Sequence[tuple[str, list[str], int]], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    users = [row[0] for row in batch]
    histories = [row[1] for row in batch]
    raw_scores = evaluator.score_batch(users, histories)
    scores = np.stack([normalize_score_rows(score) for score in raw_scores], axis=1)
    score_tensor = torch.as_tensor(scores, dtype=torch.float32, device=device)
    history_lengths = torch.as_tensor(
        [len(row[1]) for row in batch], dtype=torch.float32, device=device
    )
    targets = torch.as_tensor([row[2] for row in batch], dtype=torch.long, device=device)
    return score_tensor, history_lengths, targets


def gate_features(scores: torch.Tensor, history_lengths: torch.Tensor) -> torch.Tensor:
    """Build inference-time features from normalized [batch, expert, item] scores."""
    top2 = torch.topk(scores, k=2, dim=2).values
    means = scores.mean(dim=2)
    stds = scores.std(dim=2)
    confidence = top2[:, :, 0] - means
    margins = top2[:, :, 0] - top2[:, :, 1]
    length = torch.log1p(history_lengths).unsqueeze(1) / 5.0
    return torch.cat([length, confidence, margins, stds], dim=1)


class AdaptiveGate(torch.nn.Module):
    def __init__(self, feature_dim: int, initial_weights: Sequence[float]):
        super().__init__()
        self.linear = torch.nn.Linear(feature_dim, 3)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(torch.log(torch.tensor(initial_weights)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(features), dim=1)


def metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    target_scores = scores[np.arange(len(targets)), targets]
    ranks = (scores > target_scores[:, None]).sum(axis=1)
    result: dict[str, float] = {}
    for k in (5, 10, 20):
        hits = ranks < k
        result[f"R@{k}"] = float(hits.mean())
        result[f"N@{k}"] = float(np.where(hits, 1.0 / np.log2(ranks + 2), 0.0).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sasrec-model", required=True)
    parser.add_argument("--itemcf-model", required=True)
    parser.add_argument("--semantic-emb", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--score-scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-weights", default="0.65,0.15,0.20")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.score_scale <= 0:
        parser.error("batch-size, epochs, and score-scale must be positive")
    initial_weights = [float(value) for value in args.initial_weights.split(",")]
    if len(initial_weights) != 3 or any(value <= 0 for value in initial_weights):
        parser.error("initial-weights must contain three positive comma-separated values")

    torch.manual_seed(args.seed)
    evaluator = ExpertEvaluator.from_paths(
        args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb,
        device=args.device, mask_history=not args.no_mask,
    )
    val_examples = collect_examples(evaluator, "val")
    test_examples = collect_examples(evaluator, "test")
    print(f"[data] {evaluator.dataset.name}: val={len(val_examples)} test={len(test_examples)}", flush=True)

    gate = AdaptiveGate(10, initial_weights).to(args.device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.l2)
    for epoch in range(1, args.epochs + 1):
        gate.train()
        losses: list[float] = []
        for start in tqdm(range(0, len(val_examples), args.batch_size), desc=f"train-{epoch}", unit="batch"):
            batch = val_examples[start : start + args.batch_size]
            scores, lengths, targets = batch_scores(evaluator, batch, args.device)
            weights = gate(gate_features(scores, lengths))
            fused = (weights.unsqueeze(2) * scores).sum(dim=1)
            loss = torch.nn.functional.cross_entropy(args.score_scale * fused, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[train] epoch={epoch} cross_entropy={np.mean(losses):.5f}", flush=True)

    gate.eval()
    all_scores: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(test_examples), args.batch_size), desc="test", unit="batch"):
            batch = test_examples[start : start + args.batch_size]
            scores, lengths, targets = batch_scores(evaluator, batch, args.device)
            weights = gate(gate_features(scores, lengths))
            fused = (weights.unsqueeze(2) * scores).sum(dim=1)
            all_scores.append(fused.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
    test_scores = np.concatenate(all_scores)
    test_targets = np.concatenate(all_targets)
    test_weights = np.concatenate(all_weights)
    summary = metrics(test_scores, test_targets)
    output = {
        "dataset": evaluator.dataset.name,
        "protocol": {
            "gate_fit_split": "validation",
            "test_used_for_training_or_selection": False,
            "mask_history": not args.no_mask,
            "ranking": "full_catalog",
        },
        "hyperparameters": {
            "epochs": args.epochs, "lr": args.lr, "l2": args.l2,
            "score_scale": args.score_scale, "seed": args.seed,
            "initial_weights": initial_weights,
        },
        "test_metrics": summary,
        "mean_test_weights": test_weights.mean(axis=0).tolist(),
        "std_test_weights": test_weights.std(axis=0).tolist(),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(f"[test] {json.dumps(summary, sort_keys=True)}", flush=True)
    print(f"[test] mean_weights={output['mean_test_weights']}", flush=True)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
