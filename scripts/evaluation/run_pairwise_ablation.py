"""Pairwise ablation for RQ2: report all 7 ablation rows in one pass.

Rows:
  1. SASRec only         (1, 0, 0)
  2. ItemCF only         (0, 1, 0)
  3. Semantic only       (0, 0, 1)
  4. SASRec + ItemCF     (w_seq, w_cf, 0)   renormalized
  5. SASRec + Semantic   (w_seq, 0, w_sem)  renormalized
  6. ItemCF + Semantic   (0, w_cf, w_sem)   renormalized
  7. 3-expert            (w_seq, w_cf, w_sem)  val-selected from gate_selection JSON

Base weights are loaded from outputs/gate_selection_{dataset}_{encoder}.json so the
pairwise rows use the same per-expert weight as the val-selected 3-expert row.

Usage:
  python3 -m scripts.evaluation.run_pairwise_ablation --dataset amazon_beauty --encoder bge_base
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lime_rec.evaluation import ExpertEvaluator, normalize_scores


DATASETS = {
    "amazon_beauty": {
        "config": "configs/amazon_beauty.json",
        "sasrec": "outputs/models/amazon_beauty_sasrec.pt",
        "itemcf": "outputs/models/amazon_beauty_itemcf.json",
    },
    "amazon_toys": {
        "config": "configs/amazon_toys.json",
        "sasrec": "outputs/models/amazon_toys_sasrec.pt",
        "itemcf": "outputs/models/amazon_toys_itemcf.json",
    },
    "amazon_sports": {
        "config": "configs/amazon_sports.json",
        "sasrec": "outputs/models/amazon_sports_sasrec.pt",
        "itemcf": "outputs/models/amazon_sports_itemcf.json",
    },
    "amazon_music": {
        "config": "configs/amazon_music.json",
        "sasrec": "outputs/models/amazon_music_sasrec.pt",
        "itemcf": "outputs/models/amazon_music_itemcf.json",
    },
}


def renorm(triple):
    s = sum(triple)
    if s == 0:
        return (0.0, 0.0, 0.0)
    return tuple(x / s for x in triple)


def evaluate(dataset_name, encoder, ks=(5, 10, 20)):
    paths = DATASETS[dataset_name]
    sem_path = f"outputs/embeddings/{dataset_name}_{encoder}.npz"
    gate_path = f"outputs/gate_selection_{dataset_name}_{encoder}.json"
    if encoder == "minilm":
        gate_alt = f"outputs/gate_selection_{dataset_name}.json"
        if not Path(gate_path).exists() and Path(gate_alt).exists():
            gate_path = gate_alt

    gate = json.loads(Path(gate_path).read_text())
    sel = gate.get("selected", gate)
    w_seq = float(sel["w_seq"])
    w_cf = float(sel["w_cf"])
    w_sem = float(sel["w_sem"])

    evaluator = ExpertEvaluator.from_paths(
        paths["config"], paths["sasrec"], paths["itemcf"], sem_path
    )
    ds = evaluator.dataset
    print(f"[{dataset_name}] users={ds.num_users} items={ds.num_items} "
          f"encoder={encoder} val-selected=({w_seq:.3f},{w_cf:.3f},{w_sem:.3f})", flush=True)

    item_index = {iid: i for i, iid in enumerate(ds.item_ids)}
    rows = {
        "SASRec_only":       renorm((w_seq, 0.0, 0.0)),
        "ItemCF_only":       renorm((0.0, w_cf, 0.0)),
        "Semantic_only":     renorm((0.0, 0.0, w_sem)),
        "SASRec+ItemCF":     renorm((w_seq, w_cf, 0.0)),
        "SASRec+Semantic":   renorm((w_seq, 0.0, w_sem)),
        "ItemCF+Semantic":   renorm((0.0, w_cf, w_sem)),
        "3-expert":          (w_seq, w_cf, w_sem),
    }

    metrics = {row: {f"{m}@{k}": [] for m in ("R", "N") for k in ks} for row in rows}

    for u in tqdm(ds.user_ids, desc=f"{dataset_name}-test", unit="user"):
        history = ds.history_by_user.get(u, []) + [ds.valid_by_user.get(u, "")]
        history = [h for h in history if h]
        target_item = ds.test_by_user.get(u, "")
        if not history or not target_item:
            continue
        target = item_index.get(target_item)
        if target is None:
            continue

        sa, ci, se = evaluator.score(u, history)
        sa_n, ci_n, se_n = map(normalize_scores, (sa, ci, se))

        for row_name, (a, b, c) in rows.items():
            scores = a * sa_n + b * ci_n + c * se_n
            rank = int((scores > scores[target]).sum())
            for k in ks:
                hit = 1.0 if rank < k else 0.0
                ndcg = 1.0 / math.log2(rank + 2) if rank < k else 0.0
                metrics[row_name][f"R@{k}"].append(hit)
                metrics[row_name][f"N@{k}"].append(ndcg)

    summary = {row: {k: float(np.mean(v)) for k, v in d.items()} for row, d in metrics.items()}
    return summary, {"w_seq": w_seq, "w_cf": w_cf, "w_sem": w_sem}, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--encoder", default="bge_base")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    summary, base_weights, applied = evaluate(args.dataset, args.encoder)
    out = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "base_val_selected_weights": base_weights,
        "applied_weights_per_row": {k: list(v) for k, v in applied.items()},
        "results": summary,
    }
    out_path = args.out or f"outputs/pairwise_ablation_{args.dataset}_{args.encoder}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")

    print(f"\n{'Row':<22}{'R@10':>10}{'N@10':>10}")
    print("-" * 42)
    for row, m in summary.items():
        print(f"{row:<22}{m['R@10']:>10.4f}{m['N@10']:>10.4f}")


if __name__ == "__main__":
    main()
