"""Summarize three-seed controlled reports after all alignment gates pass."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output_final/results/controlled_gram")
    args = parser.parse_args()
    root = Path(args.root)
    alignment = list((root / "alignment").glob("amazon_*.json"))
    if len(alignment) != 3 or not all(json.loads(path.read_text()).get("passed") for path in alignment):
        raise RuntimeError("all three alignment audits must pass before summary")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for model_dir in ("lime_matched20", "gram"):
        for path in (root / model_dir).glob("amazon_*/seed[0-9].json"):
            row = json.loads(path.read_text())
            grouped[(row["dataset"], row["model"])].append(row)
    summary_rows = []
    for (dataset, model), reports in sorted(grouped.items()):
        seeds = sorted(report["seed"] for report in reports)
        if seeds != [0, 1, 2]:
            raise RuntimeError(f"{dataset}/{model} requires exactly seeds 0,1,2")
        row = {"Dataset": dataset, "Model": model}
        for metric in ("R@5", "N@5", "R@10", "N@10"):
            values = np.asarray([report["metrics"][metric] for report in reports])
            row[metric] = f"{values.mean():.6f} +/- {values.std(ddof=1):.6f}"
        summary_rows.append(row)
    payload = {"controlled_protocol": "GRAM max_his=20 vs LIME-Rec matched20",
               "seeds": [0, 1, 2], "rows": summary_rows,
               "native_lime_note": "LIME-Rec Native is a different protocol and is excluded."}
    (root / "controlled_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (root / "controlled_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Dataset", "Model", "R@5", "N@5", "R@10", "N@10"])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
