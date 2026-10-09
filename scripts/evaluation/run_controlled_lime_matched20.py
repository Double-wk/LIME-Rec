"""Fit and evaluate LIME-Rec under the controlled max-history-20 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lime_rec.controlled_metrics import evaluate_rankings
from lime_rec.baselines.gram import sha256_file
from lime_rec.evaluation import ExpertEvaluator
from lime_rec.protocols import truncate_history
from scripts.evaluation.run_repeat_aware_gate import (
    RepeatAwareGate,
    batch_scores,
    collect_examples,
    features,
)


HISTORY_CAP = 20


def _controlled_examples(evaluator: ExpertEvaluator, split: str):
    return [(user, truncate_history(history, HISTORY_CAP), target)
            for user, history, target in collect_examples(evaluator, split)]


def _ranking(scores: np.ndarray, item_ids: list[str], k: int) -> list[str]:
    order = sorted(range(len(item_ids)), key=lambda i: (-float(scores[i]), item_ids[i]))[:k]
    return [item_ids[index] for index in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sasrec-model", required=True)
    parser.add_argument("--itemcf-model", required=True)
    parser.add_argument("--semantic-emb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--score-scale", type=float, default=20.0)
    parser.add_argument("--max-penalty", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-weights", default="0.60,0.15,0.25")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--disable-calibration", action="store_true",
                        help="Force the bounded history penalty to zero in both gate "
                             "fitting and evaluation (no-calibration recovery witness).")
    parser.add_argument("--mask-history", action="store_true",
                        help="Exclude history-window items from candidates "
                             "(repeat-masked protocol variant; default keeps them).")
    parser.add_argument("--eval-mask-history", action="store_true",
                        help="Fit the gate on unmasked scores (repeat-allowed) but "
                             "evaluate with history-masked scores; frozen-weight "
                             "sensitivity diagnostic.")
    parser.add_argument("--experts", default="seq,cf,sem",
                        help="Comma subset of {seq,cf,sem}; unselected experts get zero "
                             "fusion weight (e.g. seq,cf = no-semantic witness).")
    parser.add_argument("--model-label", default="LIME-Rec-matched20")
    args = parser.parse_args()
    initial = tuple(float(value) for value in args.initial_weights.split(","))
    if len(initial) != 3 or any(value <= 0 for value in initial):
        parser.error("initial-weights must contain three positive values")
    if args.top_k < 10:
        parser.error("top-k must be at least 10")
    expert_names = ["seq", "cf", "sem"]
    selected = [token.strip() for token in args.experts.split(",") if token.strip()]
    if not selected or any(token not in expert_names for token in selected):
        parser.error("experts must be a non-empty subset of {seq,cf,sem}")
    expert_mask = torch.tensor(
        [1.0 if name in selected else 0.0 for name in expert_names],
        dtype=torch.float32, device=args.device,
    )

    def combine(scores, lengths, targets, history):
        weights, penalty = gate(features(scores, lengths, history))
        weights = weights * expert_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if args.disable_calibration:
            penalty = torch.zeros_like(penalty)
        return (weights.unsqueeze(2) * scores).sum(1) - penalty.unsqueeze(1) * history

    torch.manual_seed(args.seed)
    evaluator = ExpertEvaluator.from_paths(
        args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb,
        device=args.device, mask_history=args.mask_history,
    )
    test_evaluator = evaluator
    if args.eval_mask_history:
        if args.mask_history:
            parser.error("--eval-mask-history and --mask-history are mutually exclusive")
        test_evaluator = ExpertEvaluator.from_paths(
            args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb,
            device=args.device, mask_history=True,
        )
    if evaluator.sasrec.maxlen != HISTORY_CAP:
        raise RuntimeError(
            f"matched-20 requires a retrained maxlen=20 checkpoint; got maxlen={evaluator.sasrec.maxlen}"
        )
    validation = _controlled_examples(evaluator, "val")
    test = _controlled_examples(evaluator, "test")
    if test_evaluator is not evaluator:
        # Rebuild test examples against the masking evaluator so batch_scores
        # applies the history mask at evaluation time only.
        test = _controlled_examples(test_evaluator, "test")
    gate = RepeatAwareGate(14, initial, args.max_penalty).to(args.device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.l2)

    for epoch in range(args.epochs):
        gate.train()
        for start in tqdm(range(0, len(validation), args.batch_size), desc=f"fit-{epoch + 1}"):
            scores, lengths, targets, history = batch_scores(
                evaluator, validation[start:start + args.batch_size], args.device)
            fused = combine(scores, lengths, targets, history)
            loss = torch.nn.functional.cross_entropy(args.score_scale * fused, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    rankings: dict[str, list[str]] = {}
    targets_by_user: dict[str, str] = {}
    rows: list[dict] = []
    gate.eval()
    all_weights: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(test), args.batch_size), desc="test"):
            batch = test[start:start + args.batch_size]
            scores, lengths, targets, history = batch_scores(test_evaluator, batch, args.device)
            weights, _ = gate(features(scores, lengths, history))
            weights = weights * expert_mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            all_weights.append(weights.cpu().numpy())
            fused = combine(scores, lengths, targets, history).cpu().numpy()
            for row_index, (user_id, _, target_index) in enumerate(batch):
                ranking = _ranking(fused[row_index], evaluator.dataset.item_ids, args.top_k)
                target_id = evaluator.dataset.item_ids[target_index]
                rankings[user_id] = ranking
                targets_by_user[user_id] = target_id
                rows.append({"user_id": user_id, "target_item_id": target_id,
                             "ranking": ranking, "seed": args.seed,
                             "dataset": evaluator.dataset.name,
                             "model": args.model_label})

    metrics = evaluate_rankings(rankings, targets_by_user, ks=(5, 10))
    output = {
        "dataset": evaluator.dataset.name,
        "model": args.model_label,
        "seed": args.seed,
        "protocol": {"history_cap": HISTORY_CAP, "sasrec_maxlen": evaluator.sasrec.maxlen,
                     "gate_fit_split": "validation", "mask_history": bool(args.mask_history),
                     "eval_mask_history": bool(args.eval_mask_history),
                     "candidate_universe": "full_catalog", "ranking_export_k": args.top_k,
                     "disable_calibration": bool(args.disable_calibration),
                     "experts": selected,
                     "test_used_for_training_or_selection": False},
        "metrics": metrics,
        "mean_test_weights": np.concatenate(all_weights).mean(axis=0).tolist(),
        "artifacts": {name: {"path": path, "sha256": sha256_file(path)} for name, path in
                      {"sasrec": args.sasrec_model, "itemcf": args.itemcf_model,
                       "semantic": args.semantic_emb}.items()},
    }
    out_path = Path(args.out)
    pred_path = Path(args.predictions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
