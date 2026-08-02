"""Evaluate a single SHARED weight triple (w_seq, w_cf, w_sem) across multiple datasets.

This generates the "supplementary canonical table" showing LIME-Rec's gains do not
require per-dataset weight tuning. Each dataset is evaluated on its own test set
with the SAME global weights.

Usage:
  python3 -m scripts.evaluation.run_shared_weight \
    --weights 0.50,0.20,0.30 \
    --out outputs/3expert_shared_weight_bts.json

Derives dataset/model/embedding paths from the dataset name. A configured
dataset therefore cannot be silently skipped just because it was not added to
a hard-coded evaluation map.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lime_rec.evaluation import ExpertEvaluator, normalize_score_rows


def paths_for_dataset(name: str, encoder: str, sasrec_suffix: str) -> dict[str, str]:
    """Resolve the standard artifact paths for one configured dataset."""
    config = Path("configs") / f"{name}.json"
    if not config.is_file():
        raise ValueError(
            f"Unknown dataset {name!r}: expected configuration file {config}."
        )

    suffix = f"_{sasrec_suffix}" if sasrec_suffix else ""
    return {
        "config": str(config),
        "sasrec": f"outputs/models/{name}_sasrec{suffix}.pt",
        "itemcf": f"outputs/models/{name}_itemcf.json",
        "semantic": f"outputs/embeddings/{name}_{encoder}.npz",
    }


def evaluate_dataset(name, paths, w_seq, w_cf, w_sem, ks=(5, 10, 20)):
    evaluator = ExpertEvaluator.from_paths(
        paths["config"], paths["sasrec"], paths["itemcf"], paths["semantic"],
        device=DEVICE, mask_history=MASK_HISTORY
    )
    ds = evaluator.dataset
    print(f"[{name}] users={ds.num_users} items={ds.num_items}", flush=True)

    item_index = {iid: i for i, iid in enumerate(ds.item_ids)}
    metrics = {f"{k}@{kk}": [] for k in ("R", "N") for kk in ks}
    metrics["SASRec"] = {f"{k}@{kk}": [] for k in ("R", "N") for kk in ks}
    metrics["3expert_shared"] = {f"{k}@{kk}": [] for k in ("R", "N") for kk in ks}

    examples = []
    for u in ds.user_ids:
        history = [
            h for h in ds.history_by_user.get(u, []) + [ds.valid_by_user.get(u, "")] if h
        ]
        target = item_index.get(ds.test_by_user.get(u, ""))
        if history and target is not None:
            examples.append((u, history, target))

    for start in tqdm(range(0, len(examples), BATCH_SIZE), desc=f"{name}-test", unit="batch"):
        batch = examples[start : start + BATCH_SIZE]
        users = [row[0] for row in batch]
        histories = [row[1] for row in batch]
        targets = np.asarray([row[2] for row in batch], dtype=np.int64)
        sa, ci, se = evaluator.score_batch(users, histories)
        sa_n, ci_n, se_n = map(normalize_score_rows, (sa, ci, se))
        fused = w_seq * sa_n + w_cf * ci_n + w_sem * se_n

        for method_name, scores in [("SASRec", sa), ("3expert_shared", fused)]:
            target_scores = scores[np.arange(len(batch)), targets]
            ranks = (scores > target_scores[:, None]).sum(axis=1)
            for k in ks:
                hits = ranks < k
                ndcgs = np.where(hits, 1.0 / np.log2(ranks + 2), 0.0)
                metrics[method_name][f"R@{k}"].extend(hits.astype(float))
                metrics[method_name][f"N@{k}"].extend(ndcgs)

    summary = {}
    for method_name in ["SASRec", "3expert_shared"]:
        summary[method_name] = {k: float(np.mean(v)) for k, v in metrics[method_name].items()}
    return summary


def main():
    global DEVICE, MASK_HISTORY, BATCH_SIZE
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="0.50,0.20,0.30",
                   help="Shared weight triple as w_seq,w_cf,w_sem")
    p.add_argument("--datasets", default="amazon_beauty,amazon_toys,amazon_sports",
                   help="Comma list of dataset names to run")
    p.add_argument("--encoder", default="minilm", help="Encoder name; resolves outputs/embeddings/{ds}_{encoder}.npz")
    p.add_argument("--sasrec-suffix", default="", help="Optional SASRec checkpoint suffix, e.g. 'bce' resolves to {ds}_sasrec_bce.pt")
    p.add_argument("--out", default=None, help="Default: outputs/3expert_shared_weight_bts_{encoder}.json")
    p.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="inference device")
    p.add_argument("--no-mask", action="store_true", help="do not exclude interacted history items")
    p.add_argument("--batch-size", type=int, default=256, help="users per inference batch")
    args = p.parse_args()
    DEVICE = args.device
    MASK_HISTORY = not args.no_mask
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    BATCH_SIZE = args.batch_size

    try:
        weights = [float(value) for value in args.weights.split(",")]
    except ValueError as exc:
        p.error(f"--weights must contain three comma-separated numbers: {exc}")
    if len(weights) != 3:
        p.error("--weights must contain exactly three comma-separated values.")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        p.error("--weights must be non-negative and have a positive total.")
    w_seq, w_cf, w_sem = weights
    print(f"[shared weights] w_seq={w_seq}, w_cf={w_cf}, w_sem={w_sem}  encoder={args.encoder}", flush=True)

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not dataset_names:
        p.error("--datasets must contain at least one dataset name.")

    results = {}
    for name in dataset_names:
        try:
            paths = paths_for_dataset(name, args.encoder, args.sasrec_suffix)
        except ValueError as exc:
            p.error(str(exc))
        missing = [path for path in paths.values() if not Path(path).is_file()]
        if missing:
            p.error(
                f"{name!r} is missing required artifact(s): "
                + ", ".join(missing)
            )
        results[name] = evaluate_dataset(name, paths, w_seq, w_cf, w_sem)

    out = {
        "shared_weights": {"w_seq": w_seq, "w_cf": w_cf, "w_sem": w_sem},
        "encoder": args.encoder,
        "sasrec_suffix": args.sasrec_suffix or "default(ce)",
        "protocol": {"mask_history": MASK_HISTORY, "device": DEVICE, "ranking": "full_catalog", "batch_size": BATCH_SIZE},
        "results": results,
    }
    suffix_tag = f"_{args.sasrec_suffix}" if args.sasrec_suffix else ""
    out_path = args.out or f"outputs/3expert_shared_weight_bts_{args.encoder}{suffix_tag}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")

    # Pretty print summary table
    print(f"\n{'Dataset':<20}{'SASRec R@10':>15}{'Shared R@10':>15}{'Δ':>10}")
    print("-" * 60)
    for name, s in results.items():
        sas = s["SASRec"]["R@10"]
        fused = s["3expert_shared"]["R@10"]
        delta = (fused - sas) / sas * 100
        print(f"{name:<20}{sas:>15.4f}{fused:>15.4f}{delta:>+9.1f}%")


if __name__ == "__main__":
    main()
