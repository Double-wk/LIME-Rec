"""Recovery-threshold figure data + TikZ-free matplotlib plot.

Reads the recovery battery aggregation and writes:
  recovery_thresholds.pdf  — per-dataset seed-wise r_min scatter + grid lines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", default="output_final/results/recovery_battery/table_recovery_decomposition.json")
    parser.add_argument("--out", default="figures/recovery_thresholds.pdf")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    battery = json.loads(Path(args.battery).read_text())
    full = battery["variants"]["full"]
    datasets = list(full.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 2.6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    grid = battery["grid"]
    for ax, ds in zip(axes, datasets):
        block = full[ds]
        xs = [r["seed"] for r in block["per_seed"]]
        ys = [100.0 * r["r_min_relative"] for r in block["per_seed"]]
        ax.scatter(xs, ys, zorder=3, color="#1f77b4", s=36)
        for eta in grid:
            ax.axhline(100 * eta, ls=":", lw=0.8, color="gray")
            ax.text(2.35, 100 * eta, f"{100 * eta:g}%", fontsize=7, va="bottom", color="gray")
        ax.set_title(ds.replace("amazon_", ""))
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel("seed")
        ax.set_ylim(bottom=-0.5)
    axes[0].set_ylabel(r"$r_{\min}$ (%)")
    fig.suptitle("Seed-wise minimum supported relative tolerance (full witness)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
