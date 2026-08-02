"""Singleton / pairwise / full ablation under the repeat-aware, no-mask protocol.

For a chosen subset of experts (seq, cf, sem), fits a per-user gate over that
subset plus the same bounded history-calibration term used by the main method,
on validation only, and evaluates once on test with full-catalog ranking and no
history masking. This makes every ablation row (singletons, pairs, all three)
apples-to-apples with tab:main.

For a single-expert subset the gate is trivial (weight 1); the bounded penalty is
still learned so the protocol matches the full model.  Multi-expert subsets
inherit the formal three-expert initialization over their selected experts and
renormalize it, rather than silently switching to a uniform initialization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lime_rec.evaluation import ExpertEvaluator, normalize_score_rows
from scripts.evaluation.run_repeat_aware_gate import (
    batch_scores,
    collect_examples,
    compute_metrics,
    features,
)

EXPERTS = ("seq", "cf", "sem")


class SubsetGate(torch.nn.Module):
    """Per-user gate over a subset of experts + one bounded penalty scalar-per-user."""

    def __init__(self, feature_dim: int, initial_weights, max_penalty: float):
        super().__init__()
        self.n_sub = len(initial_weights)
        self.linear = torch.nn.Linear(feature_dim, self.n_sub + 1)
        self.max_penalty = max_penalty
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias[:self.n_sub].copy_(torch.log(torch.tensor(initial_weights)))
            self.linear.bias[self.n_sub] = -2.0

    def forward(self, x):
        out = self.linear(x)
        w = torch.softmax(out[:, : self.n_sub], dim=1)
        pen = self.max_penalty * torch.sigmoid(out[:, self.n_sub])
        return w, pen


def subset_features(scores_sub, lengths, history):
    """Same feature construction as the main gate, but over the selected experts."""
    return features(scores_sub, lengths, history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sasrec-model", required=True)
    p.add_argument("--itemcf-model", required=True)
    p.add_argument("--semantic-emb", required=True)
    p.add_argument("--experts", required=True,
                   help="comma list from {seq,cf,sem}, e.g. seq,sem")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--l2", type=float, default=0.001)
    p.add_argument("--score-scale", type=float, default=20.0)
    p.add_argument("--max-penalty", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial-weights", default="0.60,0.15,0.25",
                   help="Formal three-expert initialization in seq,cf,sem order.")
    args = p.parse_args()

    sub = [e.strip() for e in args.experts.split(",") if e.strip()]
    idx = [EXPERTS.index(e) for e in sub]
    if not idx:
        p.error("must select at least one expert")
    initial_all = [float(value) for value in args.initial_weights.split(",")]
    if len(initial_all) != len(EXPERTS) or any(value <= 0 for value in initial_all):
        p.error("initial-weights must contain three positive values in seq,cf,sem order")
    initial_sub = [initial_all[index] for index in idx]
    initial_total = sum(initial_sub)
    initial_sub = [value / initial_total for value in initial_sub]

    torch.manual_seed(args.seed)
    ev = ExpertEvaluator.from_paths(
        args.config, args.sasrec_model, args.itemcf_model, args.semantic_emb,
        device=args.device, mask_history=False,
    )
    val = collect_examples(ev, "val")
    test = collect_examples(ev, "test")
    print(f"[data] {ev.dataset.name}: val={len(val)} test={len(test)} experts={sub}", flush=True)

    sel = torch.as_tensor(idx, dtype=torch.long, device=args.device)
    feat_dim = 5 * len(idx) + 1  # length + 4 per-expert feats*? -> matched to features() output below
    # features() returns: length(1) + confidence(n) + margins(n) + stds(n) + contamination(n) + history_ratio(1)
    feat_dim = 2 + 4 * len(idx)
    gate = SubsetGate(feat_dim, initial_sub, args.max_penalty).to(args.device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.l2)

    def run_batch(batch):
        scores, lengths, targets, history = batch_scores(ev, batch, args.device)
        scores_sub = scores[:, sel, :]
        feats = subset_features(scores_sub, lengths, history)
        w, pen = gate(feats)
        fused = (w.unsqueeze(2) * scores_sub).sum(dim=1) - pen.unsqueeze(1) * history
        return fused, targets

    for epoch in range(1, args.epochs + 1):
        gate.train(); losses = []
        for s in tqdm(range(0, len(val), args.batch_size), desc=f"train-{epoch}", unit="batch"):
            fused, targets = run_batch(val[s:s + args.batch_size])
            loss = torch.nn.functional.cross_entropy(args.score_scale * fused, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[train] epoch={epoch} ce={np.mean(losses):.5f}", flush=True)

    gate.eval(); all_s, all_t = [], []
    with torch.no_grad():
        for s in tqdm(range(0, len(test), args.batch_size), desc="test", unit="batch"):
            fused, targets = run_batch(test[s:s + args.batch_size])
            all_s.append(fused.cpu().numpy()); all_t.append(targets.cpu().numpy())

    out = {
        "dataset": ev.dataset.name, "seed": args.seed, "experts": sub,
        "protocol": {"gate_fit_split": "validation",
                     "test_used_for_training_or_selection": False,
                     "mask_history": False, "ranking": "full_catalog"},
        "hyperparameters": {**vars(args), "subset_initial_weights": initial_sub},
        "test_metrics": compute_metrics(np.concatenate(all_s), np.concatenate(all_t)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[test] {json.dumps(out['test_metrics'], sort_keys=True)}", flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
