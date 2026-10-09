"""Aggregate the recovery-audit battery into manuscript-ready tables.

Inputs: recovery audit JSONs produced by scripts.evaluation.recovery_audit_gram
(single statistical pipeline; per-seed delta, LCB95, r_min, grid decisions).

Outputs:
  table_recovery_decomposition.json/.tex  — per-dataset witness variants
  recovery_thresholds.json                — per-seed r_min + grid (figure data)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GRID = (0.01, 0.025, 0.05, 0.10)
VARIANTS = ("full", "nocal", "nosem", "minilm")  # witness decomposition rows
DATASETS = ("amazon_beauty", "amazon_toys", "amazon_sports")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output_final/results/controlled_gram")
    parser.add_argument("--out-dir", default="output_final/results/recovery_battery")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_files = {
        "full": root / "recovery_audit_gram.json",
        "nocal": root / "recovery_audit_gram_nocal.json",
        "nosem": root / "recovery_audit_gram_nosem.json",
        "minilm": root / "recovery_audit_gram_minilm.json",
        "native_mask_init": root / "recovery_audit_native_mask_init.json",
        "native_mask_refit": root / "recovery_audit_native_mask_refit.json",
        "native_mask_frozen": root / "recovery_audit_native_mask_frozen.json",
        "tiger": root / "recovery_audit_tiger.json",
    }
    audits = {name: _load(path) for name, path in audit_files.items()}

    table = {"datasets": {}, "grid": list(GRID), "variants": {}}
    for name, audit in audits.items():
        if audit is None:
            continue
        for ds, rows in audit["datasets"].items():
            per_seed = []
            for r in rows:
                per_seed.append({
                    "seed": r["seed"], "U_A": r["U_A"], "delta": r["delta"],
                    "lcb95": r["lcb95"], "r_min_relative": r["r_min_relative"],
                    "superiority": r["superiority"],
                    "recovery_at_margin": r["recovery_at_margin"],
                })
            max_rmin = max(r["r_min_relative"] for r in per_seed)
            all_seed_grid = {f"{eta:g}": all(r["recovery_at_margin"][f"{eta:g}"] for r in per_seed)
                             for eta in GRID}
            first_grid = next((eta for eta in GRID if all_seed_grid[f"{eta:g}"]), None)
            table["variants"].setdefault(name, {})[ds] = {
                "per_seed": per_seed,
                "max_seed_r_min": max_rmin,
                "all_seed_recovery_grid": all_seed_grid,
                "first_supported_grid_point": first_grid,
            }

    (out_dir / "table_recovery_decomposition.json").write_text(json.dumps(table, indent=2) + "\n")
    for name, datasets in table["variants"].items():
        for ds, block in datasets.items():
            seeds = " ".join(f"seed{r['seed']}: d={r['delta']:+.4f} lcb={r['lcb95']:+.4f} "
                             f"r_min={r['r_min_relative']*100:.1f}%" for r in block["per_seed"])
            print(f"[{name:>11}] {ds:<15} {seeds} | max_r_min={block['max_seed_r_min']*100:.1f}% "
                  f"first_grid={block['first_supported_grid_point']}")

    # --- LaTeX emission -----------------------------------------------------
    label = {"full": "Full witness", "nocal": "$-$ history calibration",
             "nosem": "$-$ semantic expert", "minilm": "MiniLM prior (BGE $\\rightarrow$ MiniLM)",
             "native_mask_init": "Native repeat-masked, init-gate witness",
             "native_mask_refit": "Native repeat-masked, refit-gate witness",
             "native_mask_frozen": "Native repeat-masked, frozen-gate witness",
             "tiger": "vs TIGER (second target)"}
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Audit & Dataset & $\Delta$ per seed & $\mathrm{LCB}_{95}$ per seed "
             r"& $\max_s r_{\min}^{(s)}$ & all-seed grid \\", r"\midrule"]
    for name, datasets in table["variants"].items():
        for ds, block in datasets.items():
            per = block["per_seed"]
            deltas = "/".join(f"{r['delta']:+.4f}" for r in per)
            lcbs = "/".join(f"{r['lcb95']:+.4f}" for r in per)
            maxr = f"{block['max_seed_r_min']*100:.1f}\\%"
            grid = block["first_supported_grid_point"]
            grid_s = f"{100*grid:g}\\%" if grid is not None else "none"
            lines.append(f"{label.get(name, name)} & {ds.replace('amazon_','')} & {deltas} & "
                         f"{lcbs} & {maxr} & {grid_s} \\\\")
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_recovery_decomposition.tex").write_text("\n".join(lines) + "\n")
    print(f"[saved] {out_dir / 'table_recovery_decomposition.json'}")
    print(f"[saved] {out_dir / 'table_recovery_decomposition.tex'}")


if __name__ == "__main__":
    main()
