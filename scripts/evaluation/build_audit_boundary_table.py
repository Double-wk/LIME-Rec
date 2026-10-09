"""Audit-boundary table (Table B): matched / unmatched / audited dimensions.

Static protocol facts, cross-checked against artifact metadata where possible.
"""
from __future__ import annotations

import json
from pathlib import Path

ROWS = [
    # dimension, GRAM, LIME-Rec-20 witness, status
    ("train interactions", "same splits (5-core, leave-two-out)", "same", "matched"),
    ("validation split", "same", "same", "matched"),
    ("item content fields", "title/description text", "same fields", "matched"),
    ("history window (input)", "max_his=20", "capped at 20", "matched"),
    ("candidate space", "full catalog via constrained decoding", "full catalog scoring", "matched"),
    ("evaluator / metrics", "shared per-user R@10 audit", "shared", "matched"),
    ("training seeds", "0/1/2 retrained", "0/1/2 retrained", "matched"),
    ("pretrained prior", "T5-small (+ tag-generation T5)", "frozen BGE-base", "unmatched (declared)"),
    ("serving-time LM inference", "yes (T5 encoder--decoder)", "no", "audited (excluded)"),
    ("autoregressive item-ID decoding", "yes (hierarchical IDs, beam 50)", "no", "audited (excluded)"),
    ("offline computation", "indexing, CF stats", "BGE encoding, ItemCF, SASRec", "declared (offline allowed)"),
]


def main() -> None:
    out = Path("output_final/results/recovery_battery")
    out.mkdir(parents=True, exist_ok=True)
    data = [{"dimension": d, "target_gram": g, "witness": w, "status": s} for d, g, w, s in ROWS]
    (out / "table_audit_boundary.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Dimension & GRAM (target) & LIME-Rec-20 (witness) & Status \\", r"\midrule"]
    for d, g, w, s in ROWS:
        lines.append(f"{d} & {g} & {w} & {s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_audit_boundary.tex").write_text("\n".join(lines) + "\n")
    print(f"[saved] {out}/table_audit_boundary.{{json,tex}}")


if __name__ == "__main__":
    main()
