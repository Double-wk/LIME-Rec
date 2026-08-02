"""Create a traceable recovered-user summary from formal mechanism exports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = (
    "rank_promotion_no_singleton_hit",
    "itemcf_only_hit",
    "semantic_only_hit",
    "itemcf_and_semantic_hit",
)


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _one_seed(rows: list[dict]) -> dict:
    recovered = Counter()
    for row in rows:
        experts = row["experts"]
        fusion_rank = row["fusion"]["repeat_aware"]["target_rank"]
        if fusion_rank >= 10 or experts["SASRec"]["target_rank"] < 10:
            continue
        cf_hit = experts["ItemCF"]["target_rank"] < 10
        sem_hit = experts["Semantic"]["target_rank"] < 10
        if cf_hit and sem_hit:
            category = "itemcf_and_semantic_hit"
        elif cf_hit:
            category = "itemcf_only_hit"
        elif sem_hit:
            category = "semantic_only_hit"
        else:
            category = "rank_promotion_no_singleton_hit"
        recovered[category] += 1
    total = sum(recovered.values())
    return {
        "n_users": len(rows),
        "n_recovered": total,
        "recovered_percent": 100.0 * total / len(rows) if rows else 0.0,
        "categories": {
            category: {
                "count": recovered[category],
                "percent_of_recovered": 100.0 * recovered[category] / total if total else 0.0,
            }
            for category in CATEGORIES
        },
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def _plot(report: dict, output: Path) -> None:
    # Embed TrueType (fonttype 42) instead of the matplotlib default Type 3
    # fonts, which some venue PDF checks (e.g. AAAI) reject.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    seed_zero = next(seed for seed in report["per_seed"] if seed["seed"] == 0)
    categories = list(CATEGORIES)
    labels = [
        "Rank promotion\n(no singleton hit)",
        "ItemCF only",
        "Semantic only",
        "ItemCF + Semantic",
    ]
    values = [seed_zero["summary"]["categories"][category]["percent_of_recovered"]
              for category in categories]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    bars = ax.bar(labels, values, color=["#7f8c8d", "#4c78a8", "#59a14f", "#f28e2b"])
    ax.set_ylabel("Percent of recovered users")
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.set_title("Beauty seed 0: formal repeat-aware fusion")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--figure", default="")
    args = parser.parse_args()

    per_seed = []
    for raw in args.inputs:
        path = Path(raw)
        metadata = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text())
        if metadata["protocol"].get("mask_history") is not False:
            raise ValueError(f"{path}: history masking is not disabled")
        per_seed.append({"seed": metadata["gate"]["seed"], "input": str(path),
                         "summary": _one_seed(_read_rows(path))})
    per_seed.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in per_seed] != [0, 1, 2]:
        raise ValueError("recovered-user summary requires matched seeds 0, 1, 2")

    output = {
        "dataset": "amazon_beauty",
        "definition": "Fusion@10 hit and matched SASRec@10 miss; singleton categories use the same no-mask scores.",
        "per_seed": per_seed,
        "summary": {
            "recovered_percent": _mean_std(
                [row["summary"]["recovered_percent"] for row in per_seed]
            ),
            "category_percent_of_recovered": {
                category: _mean_std([
                    row["summary"]["categories"][category]["percent_of_recovered"]
                    for row in per_seed
                ])
                for category in CATEGORIES
            },
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.figure:
        _plot(output, Path(args.figure))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
