"""Train a repeat-aware expert gate on validation and evaluate once on test.

The gate learns both the three expert weights and a bounded, user-specific
penalty for previously interacted items.  It uses only inference-time features
and never fits or selects parameters on the test split.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import rankdata
from tqdm import tqdm

from lime_rec.evaluation import ExpertEvaluator, normalize_score_rows


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the top ``k`` indices in descending score order."""
    if k >= scores.size:
        return np.argsort(-scores)
    candidates = np.argpartition(-scores, k - 1)[:k]
    return candidates[np.argsort(-scores[candidates], kind="stable")]


def _spearman_full(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-corrected Spearman correlation over the complete item catalog."""
    rank_a = rankdata(a, method="average")
    rank_b = rankdata(b, method="average")
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = float(np.sqrt((rank_a * rank_a).sum() * (rank_b * rank_b).sum()))
    return 0.0 if denominator == 0.0 else float((rank_a * rank_b).sum() / denominator)


def collect_examples(evaluator: ExpertEvaluator, split: str):
    dataset = evaluator.dataset
    item_index = evaluator.sasrec.item_index
    examples = []
    for user_id in dataset.user_ids:
        history = list(dataset.history_by_user.get(user_id, []))
        target_id = dataset.valid_by_user.get(user_id, "")
        if split == "test":
            valid_id = dataset.valid_by_user.get(user_id, "")
            if valid_id:
                history.append(valid_id)
            target_id = dataset.test_by_user.get(user_id, "")
        target = item_index.get(target_id)
        if history and target is not None:
            examples.append((user_id, history, target))
    return examples


def batch_scores(evaluator: ExpertEvaluator, batch, device: str):
    users = [row[0] for row in batch]
    histories = [row[1] for row in batch]
    raw = evaluator.score_batch(users, histories)
    scores = np.stack([normalize_score_rows(score) for score in raw], axis=1)
    score_tensor = torch.as_tensor(scores, dtype=torch.float32, device=device)
    lengths = torch.as_tensor([len(row[1]) for row in batch], dtype=torch.float32, device=device)
    targets = torch.as_tensor([row[2] for row in batch], dtype=torch.long, device=device)
    history = torch.zeros((len(batch), scores.shape[-1]), dtype=torch.bool, device=device)
    item_index = evaluator.sasrec.item_index
    for row, (_, items, _) in enumerate(batch):
        indices = [item_index[item] for item in set(items) if item in item_index]
        if indices:
            history[row, torch.as_tensor(indices, device=device)] = True
    return score_tensor, lengths, targets, history


def features(scores: torch.Tensor, lengths: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(scores, k=2, dim=2).values
    means = scores.mean(dim=2)
    stds = scores.std(dim=2)
    confidence = top2[:, :, 0] - means
    margins = top2[:, :, 0] - top2[:, :, 1]
    top5 = torch.topk(scores, k=5, dim=2).indices
    expanded_history = history.unsqueeze(1).expand(-1, scores.shape[1], -1)
    contamination = expanded_history.gather(2, top5).float().mean(dim=2)
    length = torch.log1p(lengths).unsqueeze(1) / 5.0
    history_ratio = history.float().mean(dim=1, keepdim=True)
    return torch.cat([length, confidence, margins, stds, contamination, history_ratio], dim=1)


class RepeatAwareGate(torch.nn.Module):
    def __init__(self, feature_dim: int, initial_weights: Sequence[float], max_penalty: float):
        super().__init__()
        self.linear = torch.nn.Linear(feature_dim, 4)
        self.max_penalty = max_penalty
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias[:3].copy_(torch.log(torch.tensor(initial_weights)))
            self.linear.bias[3] = -2.0

    def forward(self, x: torch.Tensor):
        output = self.linear(x)
        return torch.softmax(output[:, :3], dim=1), self.max_penalty * torch.sigmoid(output[:, 3])


def compute_metrics(scores: np.ndarray, targets: np.ndarray):
    target_scores = scores[np.arange(len(targets)), targets]
    ranks = (scores > target_scores[:, None]).sum(axis=1)
    result = {}
    for k in (5, 10, 20):
        hits = ranks < k
        result[f"R@{k}"] = float(hits.mean())
        result[f"N@{k}"] = float(np.where(hits, 1.0 / np.log2(ranks + 2), 0.0).mean())
    return result


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
    parser.add_argument("--experts", default="seq,cf,sem",
                        help="Comma subset of {seq,cf,sem}. Unselected experts get zero "
                             "fusion weight (isolation ablation).")
    parser.add_argument("--disable-calibration", action="store_true",
                        help="Force the bounded history penalty to zero, isolating the "
                             "semantic-fusion contribution from history calibration.")
    parser.add_argument("--dump-per-user", default="",
                        help="If set, write per-user SASRec/fusion target ranks as jsonl for bootstrap CI.")
    parser.add_argument("--dump-mechanism", default="",
                        help="If set, write formal per-user expert/fusion evidence for mechanism diagnostics.")
    parser.add_argument("--mechanism-topk", type=int, default=10,
                        help="Top-K lists retained in --dump-mechanism (default: 10).")
    parser.add_argument("--mechanism-max-users", type=int, default=None,
                        help="Uniformly sample at most this many test users for --dump-mechanism.")
    parser.add_argument("--mechanism-sample-seed", type=int, default=42,
                        help="Random seed used by --mechanism-max-users.")
    args = parser.parse_args()
    initial = [float(value) for value in args.initial_weights.split(",")]
    if len(initial) != 3 or any(value <= 0 for value in initial):
        parser.error("initial-weights must contain three positive values")
    expert_names = ["seq", "cf", "sem"]
    selected = [token.strip() for token in args.experts.split(",") if token.strip()]
    if not selected or any(token not in expert_names for token in selected):
        parser.error("experts must be a non-empty subset of {seq,cf,sem}")
    expert_mask = torch.tensor(
        [1.0 if name in selected else 0.0 for name in expert_names],
        dtype=torch.float32, device=args.device,
    )
    if args.batch_size < 1 or args.epochs < 1 or args.max_penalty <= 0:
        parser.error("batch-size, epochs, and max-penalty must be positive")
    if args.mechanism_topk < 1:
        parser.error("mechanism-topk must be positive")
    if args.mechanism_max_users is not None and args.mechanism_max_users < 1:
        parser.error("mechanism-max-users must be positive")

    torch.manual_seed(args.seed)
    evaluator = ExpertEvaluator.from_paths(
        args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb,
        device=args.device, mask_history=False,
    )
    validation = collect_examples(evaluator, "val")
    test = collect_examples(evaluator, "test")
    print(f"[data] {evaluator.dataset.name}: val={len(validation)} test={len(test)}", flush=True)
    mechanism_users = None
    if args.dump_mechanism:
        mechanism_users = {row[0] for row in test}
        if args.mechanism_max_users is not None and len(test) > args.mechanism_max_users:
            rng = np.random.default_rng(args.mechanism_sample_seed)
            chosen = rng.choice(len(test), size=args.mechanism_max_users, replace=False)
            mechanism_users = {test[index][0] for index in chosen.tolist()}
        print(f"[mechanism] selected={len(mechanism_users)} sample_seed="
              f"{args.mechanism_sample_seed}", flush=True)
    gate = RepeatAwareGate(14, initial, args.max_penalty).to(args.device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.l2)

    for epoch in range(1, args.epochs + 1):
        gate.train()
        losses = []
        for start in tqdm(range(0, len(validation), args.batch_size), desc=f"train-{epoch}", unit="batch"):
            scores, lengths, targets, history = batch_scores(
                evaluator, validation[start:start + args.batch_size], args.device
            )
            weights, penalty = gate(features(scores, lengths, history))
            weights = weights * expert_mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            if args.disable_calibration:
                penalty = torch.zeros_like(penalty)
            fused = (weights.unsqueeze(2) * scores).sum(dim=1) - penalty.unsqueeze(1) * history
            loss = torch.nn.functional.cross_entropy(args.score_scale * fused, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[train] epoch={epoch} cross_entropy={np.mean(losses):.5f}", flush=True)

    gate.eval()
    all_scores, all_targets, all_weights, all_penalties = [], [], [], []
    per_user_rows = []  # for optional bootstrap dump: fused + SASRec-only target ranks
    mechanism_rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(test), args.batch_size), desc="test", unit="batch"):
            batch = test[start:start + args.batch_size]
            scores, lengths, targets, history = batch_scores(evaluator, batch, args.device)
            weights, penalty = gate(features(scores, lengths, history))
            weights = weights * expert_mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            if args.disable_calibration:
                penalty = torch.zeros_like(penalty)
            fused = (weights.unsqueeze(2) * scores).sum(dim=1) - penalty.unsqueeze(1) * history
            all_scores.append(fused.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
            all_penalties.append(penalty.cpu().numpy())
            if args.dump_per_user:
                fused_np = fused.cpu().numpy()
                sas_np = scores[:, 0, :].cpu().numpy()  # SASRec-only normalized catalog scores
                cf_np = scores[:, 1, :].cpu().numpy()   # ItemCF-only
                sem_np = scores[:, 2, :].cpu().numpy()  # Semantic-only
                tgt_np = targets.cpu().numpy()
                for row, (user_id, _, _) in enumerate(batch):
                    t = int(tgt_np[row])
                    per_user_rows.append({
                        "user_id": user_id,
                        "experts": {
                            "SASRec": {"target_rank": int((sas_np[row] > sas_np[row, t]).sum())},
                            "ItemCF": {"target_rank": int((cf_np[row] > cf_np[row, t]).sum())},
                            "Semantic": {"target_rank": int((sem_np[row] > sem_np[row, t]).sum())},
                        },
                        "fusion": {"repeat_aware": {"target_rank": int((fused_np[row] > fused_np[row, t]).sum())}},
                    })
            if mechanism_users is not None:
                scores_np = scores.cpu().numpy()
                fused_np = fused.cpu().numpy()
                weights_np = weights.cpu().numpy()
                penalties_np = penalty.cpu().numpy()
                targets_np = targets.cpu().numpy()
                for row, (user_id, history_items, _) in enumerate(batch):
                    if user_id not in mechanism_users:
                        continue
                    t = int(targets_np[row])
                    sas_np, cf_np, sem_np = scores_np[row]
                    mechanism_rows.append({
                        "user_id": user_id,
                        "history_length": len(history_items),
                        "target_idx": t,
                        "experts": {
                            "SASRec": {
                                "target_rank": int((sas_np > sas_np[t]).sum()),
                                "topK": _top_k_indices(sas_np, args.mechanism_topk).tolist(),
                            },
                            "ItemCF": {
                                "target_rank": int((cf_np > cf_np[t]).sum()),
                                "topK": _top_k_indices(cf_np, args.mechanism_topk).tolist(),
                            },
                            "Semantic": {
                                "target_rank": int((sem_np > sem_np[t]).sum()),
                                "topK": _top_k_indices(sem_np, args.mechanism_topk).tolist(),
                            },
                        },
                        "fusion": {
                            "repeat_aware": {
                                "weights": weights_np[row].astype(float).tolist(),
                                "penalty": float(penalties_np[row]),
                                "target_rank": int((fused_np[row] > fused_np[row, t]).sum()),
                                "topK": _top_k_indices(fused_np[row], args.mechanism_topk).tolist(),
                            },
                        },
                        "spearman_full": {
                            "SASRec,ItemCF": _spearman_full(sas_np, cf_np),
                            "SASRec,Semantic": _spearman_full(sas_np, sem_np),
                            "ItemCF,Semantic": _spearman_full(cf_np, sem_np),
                        },
                    })
    output = {
        "dataset": evaluator.dataset.name,
        "protocol": {"gate_fit_split": "validation", "test_used_for_training_or_selection": False,
                     "mask_history": False, "ranking": "full_catalog"},
        "method": "repeat-aware expert gate",
        "hyperparameters": vars(args),
        "test_metrics": compute_metrics(np.concatenate(all_scores), np.concatenate(all_targets)),
        "mean_test_weights": np.concatenate(all_weights).mean(axis=0).tolist(),
        "mean_test_penalty": float(np.concatenate(all_penalties).mean()),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[test] {json.dumps(output['test_metrics'], sort_keys=True)}", flush=True)
    print(f"[test] mean_weights={output['mean_test_weights']} penalty={output['mean_test_penalty']:.4f}", flush=True)
    if args.dump_per_user:
        dump_path = Path(args.dump_per_user)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as fh:
            for row in per_user_rows:
                fh.write(json.dumps(row) + "\n")
        print(f"[dump] {len(per_user_rows)} per-user rows -> {dump_path}", flush=True)
    if args.dump_mechanism:
        mechanism_path = Path(args.dump_mechanism)
        mechanism_path.parent.mkdir(parents=True, exist_ok=True)
        with mechanism_path.open("w", encoding="utf-8") as fh:
            for row in mechanism_rows:
                fh.write(json.dumps(row) + "\n")
        metadata_path = mechanism_path.with_suffix(mechanism_path.suffix + ".metadata.json")
        metadata = {
            "dataset": evaluator.dataset.name,
            "source_report": str(path),
            "protocol": output["protocol"],
            "gate": {
                "initial_weights": initial,
                "score_scale": args.score_scale,
                "max_penalty": args.max_penalty,
                "seed": args.seed,
                "fit_split": "validation",
            },
            "sample": {
                "n_users": len(mechanism_rows),
                "max_users": args.mechanism_max_users,
                "sampling_seed": args.mechanism_sample_seed,
                "topk": args.mechanism_topk,
            },
            "checkpoint": args.sasrec_model,
            "itemcf": args.itemcf_model,
            "semantic_embedding": args.semantic_emb,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[mechanism] {len(mechanism_rows)} rows -> {mechanism_path}", flush=True)


if __name__ == "__main__":
    main()
