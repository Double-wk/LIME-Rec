"""Canonical recovery audit: LIME-Rec-20 vs retrained GRAM (paired bootstrap).

This is the single statistical pipeline behind the main paper's recovery
decisions (Tables 2-3): for each dataset and training seed it loads the frozen
GRAM and LIME-Rec-20 prediction files, computes the paired per-user R@10
difference, and estimates the one-sided 95% lower confidence bound with a
user-level percentile bootstrap (10,000 resamples, seed 2027). It then reports
superiority, the minimum supported relative tolerance r_min, and the recovery
decision on the prespecified sensitivity grid eta in {1%, 2.5%, 5%, 10%}.

Usage:
  python3 -m scripts.evaluation.recovery_audit_gram \
    --predictions-dir output_final/results/controlled_gram/predictions \
    --out-json output_final/results/controlled_gram/recovery_audit_gram.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRID = (0.01, 0.025, 0.05, 0.10)
DATASETS = ("amazon_beauty", "amazon_toys", "amazon_sports")
SEEDS = (0, 1, 2)


def _hits(path: Path, k: int) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ranking = row["ranking"][:k]
            out[row["user_id"]] = 1.0 if row["target_item_id"] in ranking else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--seeds", default="0,1,2",
                        help="Comma-separated training seeds to audit.")
    parser.add_argument("--gram-prefix", default="gram",
                        help="Filename prefix of the audited target predictions.")
    parser.add_argument("--lime-prefix", default="lime",
                        help="Filename prefix of the recovery witness predictions.")
    parser.add_argument("--comparison", default="LIME-Rec-20 minus retrained GRAM (per-seed paired)")
    parser.add_argument("--datasets", default=",".join(DATASETS),
                        help="Comma-separated subset of datasets to audit.")
    args = parser.parse_args()

    root = Path(args.predictions_dir)
    rng = np.random.default_rng(args.seed)
    result = {
        "criterion": ("recovery witness iff LCB95(U(R)-U(A)) > -delta; "
                      "superiority iff LCB95 > 0"),
        "comparison": args.comparison,
        "bootstrap": {"unit": "user", "resamples": args.n_resamples,
                      "seed": args.seed, "method": "percentile, one-sided lower 95%"},
        "margins_relative_to_U_A": list(GRID),
        "datasets": {},
    }

    for dataset in args.datasets.split(","):
        rows = []
        for seed in [int(x) for x in args.seeds.split(",")]:
            gram = _hits(root / f"{args.gram_prefix}_{dataset}_seed{seed}.jsonl", args.k)
            lime = _hits(root / f"{args.lime_prefix}_{dataset}_seed{seed}.jsonl", args.k)
            users = sorted(set(gram) & set(lime))
            if len(users) != len(gram) or len(users) != len(lime):
                raise RuntimeError(f"user mismatch for {dataset} seed{seed}")
            g = np.asarray([gram[u] for u in users])
            l = np.asarray([lime[u] for u in users])
            diffs = l - g
            idx = rng.integers(0, len(users), size=(args.n_resamples, len(users)))
            boot = diffs[idx].mean(axis=1)
            lcb = float(np.percentile(boot, 5.0))
            u_a = float(g.mean())
            delta = float(diffs.mean())
            r_min = max(0.0, -lcb) / u_a
            rows.append({
                "seed": seed,
                "n_users": len(users),
                "U_A": u_a,
                "delta": delta,
                "lcb95": lcb,
                "superiority": lcb > 0.0,
                "r_min_relative": r_min,
                "recovery_at_margin": {f"{eta:g}": bool(lcb > -eta * u_a) for eta in GRID},
            })
        result["datasets"][dataset] = rows
        summary = ", ".join(
            f"seed{r['seed']}: d={r['delta']:+.4f} lcb={r['lcb95']:+.4f} r_min={r['r_min_relative']*100:.1f}%"
            for r in rows)
        print(f"[{dataset}] {summary}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
