"""Regime-stratified recovery profile r_min(z) for any audited mechanism.

Reuses the audit statistics of scripts.agentic.recovery_audit_agent (same
criterion, paired user-level percentile bootstrap, grid, and per-cell RNG) and
applies them inside prespecified buckets of an observable regime variable z.
Here z is the number of interactions visible before the test recommendation
(training history plus the validation interaction), computed with
lime_rec.protocols.prediction_context so that the generative and controller
audits share one regime axis.

The bucketing is fixed in advance (default edges {4, 7}, i.e. the 5-core
minimum-history bucket, a middle bucket, and the long-history bucket) so that
the strata do not depend on the observed distribution and stay comparable across
mechanisms, datasets, and seeds. Stratum decisions are descriptive: they carry no
separate significance claim and are never used for model or protocol selection.

Usage:
  python3 -m scripts.evaluation.recovery_audit_regime \
    --predictions-dir output_final/results/controlled_gram/predictions \
    --target-prefix gram --witness-prefix lime \
    --out-json output_final/results/recovery_battery/table_regime_profile.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lime_rec.evaluation import load_configured_dataset
from lime_rec.protocols import prediction_context
from scripts.agentic.recovery_audit_agent import GRID, _audit, _buckets, _cell_rng

DATASETS = ("amazon_beauty", "amazon_toys", "amazon_sports")
SEEDS = (0, 1, 2)
CONFIGS = {"amazon_beauty": "configs/amazon_beauty.json",
           "amazon_toys": "configs/amazon_toys.json",
           "amazon_sports": "configs/amazon_sports.json"}


def _hits(path: Path, k: int) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[row["user_id"]] = 1.0 if row["target_item_id"] in row["ranking"][:k] else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--target-prefix", required=True,
                        help="Filename prefix of the audited target predictions.")
    parser.add_argument("--witness-prefix", required=True,
                        help="Filename prefix of the recovery witness predictions.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-tex")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--z-bins", default="4,7",
                        help="Prespecified bucket edges for the regime variable.")
    parser.add_argument("--label", default="audit")
    args = parser.parse_args()

    root = Path(args.predictions_dir)
    edges = [float(value) for value in args.z_bins.split(",") if value]
    result = {
        "criterion": ("recovery witness iff LCB95(U(R)-U(A)) > -delta; "
                      "superiority iff LCB95 > 0"),
        "regime_variable": ("interactions visible before the test recommendation "
                            "(training history plus the validation interaction)"),
        "z_bins": edges,
        "bootstrap": {"unit": "user", "resamples": args.n_resamples, "seed": args.seed,
                      "method": "percentile, one-sided lower 95%", "k": args.k},
        "margins_relative_to_U_A": list(GRID),
        "label": args.label,
        "datasets": {},
    }

    for dataset in [name for name in DATASETS if name in set(args.datasets.split(","))]:
        config = load_configured_dataset(CONFIGS[dataset])
        blocks = []
        for seed in SEEDS:
            target = _hits(root / f"{args.target_prefix}_{dataset}_seed{seed}.jsonl", args.k)
            witness = _hits(root / f"{args.witness_prefix}_{dataset}_seed{seed}.jsonl", args.k)
            users = sorted(set(target) & set(witness))
            if not users:
                raise RuntimeError(f"no overlapping users for {dataset} seed{seed}")
            z = np.asarray([len(prediction_context(config, user, "test").history)
                            for user in users], dtype=float)
            _edges, masks = _buckets(z, edges)
            strata = []
            for index, mask in enumerate(masks):
                subset = [user for user, keep in zip(users, mask) if keep]
                if not subset:
                    continue
                cell = _audit(np.asarray([witness[u] for u in subset]),
                              np.asarray([target[u] for u in subset]),
                              _cell_rng(args.seed, f"{dataset}:{index}:{seed}", args.label),
                              args.n_resamples)
                cell.update({"stratum": index, "n_users": len(subset),
                             "z_range": [float(z[mask].min()), float(z[mask].max())],
                             "seed": seed})
                strata.append(cell)
            block = {
                "seed": seed, "n_users": len(users),
                "overall": _audit(np.asarray([witness[u] for u in users]),
                                  np.asarray([target[u] for u in users]),
                                  _cell_rng(args.seed, f"{dataset}:overall:{seed}", args.label),
                                  args.n_resamples),
                "strata": strata,
            }
            blocks.append(block)
            profile = " ".join(
                f"q{c['stratum']}(n={c['n_users']}):d={c['delta']:+.4f},"
                f"lcb={c['lcb95']:+.4f},r_min={100*c['r_min_relative']:.1f}%"
                for c in strata)
            print(f"[{dataset} seed{seed}] overall d={block['overall']['delta']:+.4f} "
                  f"lcb={block['overall']['lcb95']:+.4f} | {profile}")
        result["datasets"][dataset] = blocks

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[saved] {out}")

    if args.out_tex:
        lines = [r"\begin{tabular}{llccccc}", r"\toprule",
                 r"Dataset & $z$ bucket & $U_A$ & $U_R$ & $\Delta$ & $\mathrm{LCB}_{95}$ "
                 r"& $r_{\min}$ \\", r"\midrule"]
        for dataset, blocks in result["datasets"].items():
            for block in blocks:
                for cell in block["strata"]:
                    lines.append(
                        f"{dataset.replace('amazon_', '')}, seed{block['seed']} & "
                        f"${cell['z_range'][0]:g}$--${cell['z_range'][1]:g}$ & "
                        f"{cell['U_A']:.3f} & {cell['U_R']:.3f} & {cell['delta']:+.4f} & "
                        f"{cell['lcb95']:+.4f} & ${100*cell['r_min_relative']:.1f}\\%$ \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        Path(args.out_tex).write_text("\n".join(lines) + "\n")
        print(f"[saved] {args.out_tex}")


if __name__ == "__main__":
    main()