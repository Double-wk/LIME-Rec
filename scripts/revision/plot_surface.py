"""Plot validation simplex diagnostics, keeping CE scales comparable."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    report = json.loads(Path(args.summary).read_text())
    grid = {k: v for k, v in report['search_log'].items() if k.startswith('grid_')}
    penalties = sorted({v['penalty'] for v in grid.values()})
    fig, axes = plt.subplots(2, len(penalties), figsize=(4 * len(penalties), 7), squeeze=False)
    for col, penalty in enumerate(penalties):
        names = [k for k, v in grid.items() if v['penalty'] == penalty]
        weights = np.array([grid[k]['weights'] for k in names])
        x = weights[:, 1] + weights[:, 2] / 2
        y = weights[:, 2] * np.sqrt(3) / 2
        for row, metric in enumerate(('R@10', 'CE')):
            values = [report['summaries']['val'][k][metric] for k in names]
            ax = axes[row, col]
            dots = ax.scatter(x, y, c=values, cmap='viridis', s=40)
            ax.set(title=f'Validation {metric}, penalty={penalty:g}', aspect='equal')
            ax.set_xticks([0, 1]); ax.set_xticklabels(['SASRec', 'ItemCF'])
            ax.set_yticks([np.sqrt(3)/2]); ax.set_yticklabels(['Semantic'])
            fig.colorbar(dots, ax=ax, shrink=.7)
    fig.tight_layout()
    dest = Path(args.out)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
