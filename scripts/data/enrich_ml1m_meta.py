"""Merge TMDB plots into ML-1M meta.jsonl as enriched semantic input.

Output: data/ml1m/meta_tmdb.jsonl  — same schema as meta.jsonl but description is
  "<title>. <genres>. <tmdb overview>. <tagline>"

Items with no TMDB hit fall back to original description (keeps coverage at 100%).
"""
from __future__ import annotations

import json
from pathlib import Path

META_IN = Path("data/ml1m/meta.jsonl")
TMDB_IN = Path("data/ml1m/tmdb_plots.jsonl")
META_OUT = Path("data/ml1m/meta_tmdb.jsonl")


def main():
    # Load TMDB plots
    plots = {}
    with open(TMDB_IN) as f:
        for line in f:
            rec = json.loads(line)
            plots[rec["item_id"]] = rec

    n_enriched = 0
    n_fallback = 0
    char_counts = []

    with open(META_IN) as f, open(META_OUT, "w") as out:
        for line in f:
            rec = json.loads(line)
            iid = rec["item_id"]
            if iid in plots and plots[iid].get("overview"):
                overview = plots[iid]["overview"]
                tagline = plots[iid].get("tagline", "")
                parts = [rec["title"], rec["category"]]
                if tagline:
                    parts.append(tagline.lower())
                parts.append(overview.lower())
                rec["description"] = ". ".join(p for p in parts if p)
                n_enriched += 1
                char_counts.append(len(rec["description"]))
            else:
                n_fallback += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    avg = sum(char_counts) / max(len(char_counts), 1)
    print(f"[enrich] {n_enriched:,} with TMDB plot, {n_fallback:,} fallback")
    print(f"[length] enriched description avg={avg:.0f} chars (was ~50)")
    print(f"[out] {META_OUT}")


if __name__ == "__main__":
    main()
