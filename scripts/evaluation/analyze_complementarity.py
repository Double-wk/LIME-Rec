"""Analyze per-user metrics JSONL to produce RQ3 complementarity tables.

Inputs: outputs/per_user_metrics_{dataset}.jsonl
        (produced by dump_per_user_metrics.py)

Outputs (printed and optionally JSON-dumped):
  RQ3.1  Jaccard@K between expert top-K sets, mean over users
  RQ3.2  Spearman rank correlation between full-catalog expert rankings
  RQ3.3  Lift attribution: among users recovered by fusion over SASRec at K=10,
         breakdown by which expert(s) had target in top-10
  RQ3.4  Hit-set decomposition: percentage of users hit by each exact subset
         of the three singleton experts at K=10

Cohort filtering (optional --cohort): split users by history length terciles
and report the same three sub-tables per cohort.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


EXPERTS = ("SASRec", "ItemCF", "Semantic")
PAIRS = [("SASRec", "ItemCF"), ("SASRec", "Semantic"), ("ItemCF", "Semantic")]


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def spearman_on_union(top_a, top_b, k_for_missing=None):
    """Spearman ρ over the union of top-K item sets.

    Items in expert A's top-K get rank position = their index in top_a (0..K-1);
    items missing from A's top-K get rank = K (tied). Same for B.
    """
    sa = list(top_a)
    sb = list(top_b)
    rank_a = {item: i for i, item in enumerate(sa)}
    rank_b = {item: i for i, item in enumerate(sb)}
    union = sorted(set(sa) | set(sb))
    if len(union) < 2:
        return 1.0
    K = max(len(sa), len(sb)) if k_for_missing is None else k_for_missing
    ra = np.array([rank_a.get(it, K) for it in union], dtype=float)
    rb = np.array([rank_b.get(it, K) for it in union], dtype=float)
    # Convert to ranks (with ties)
    ra_ranks = _avg_rank(ra)
    rb_ranks = _avg_rank(rb)
    da = ra_ranks - ra_ranks.mean()
    db = rb_ranks - rb_ranks.mean()
    denom = float(np.sqrt((da * da).sum() * (db * db).sum()))
    if denom == 0:
        return 0.0
    return float((da * db).sum() / denom)


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Average-rank assignment for arbitrary 1-D array (used by Spearman)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for kk in range(i, j + 1):
            ranks[order[kk]] = avg
        i = j + 1
    return ranks


def hit_at_k(rank: int, k: int) -> bool:
    return rank < k


def hit_set_decomposition(entries, k: int):
    """Return percentages for the eight exact singleton-hit subsets at ``k``."""
    labels = {
        (False, False, False): "NONE",
        (True, False, False): "only_SAS",
        (False, True, False): "only_ICF",
        (False, False, True): "only_SEM",
        (True, True, False): "SAS_ICF",
        (True, False, True): "SAS_SEM",
        (False, True, True): "ICF_SEM",
        (True, True, True): "ALL",
    }
    counts = Counter({label: 0 for label in labels.values()})
    for e in entries:
        ex = e["experts"]
        flags = tuple(hit_at_k(ex[name]["target_rank"], k) for name in EXPERTS)
        counts[labels[flags]] += 1
    n = len(entries)
    return {label: 100.0 * counts[label] / n for label in labels.values()}


def analyze(entries, K_jacc=10, K_lift=10):
    """Run the three sub-analyses over a list of per-user entries."""
    n = len(entries)
    if n == 0:
        return None

    # ---- RQ3.1 Jaccard ----
    jacc = {f"{a},{b}": [] for a, b in PAIRS}
    # ---- RQ3.2 Spearman (full-vector, loaded from JSONL `spearman_full` field) ----
    spear_full = {f"{a},{b}": [] for a, b in PAIRS}
    has_spear_full = "spearman_full" in entries[0]

    for e in entries:
        ex = e["experts"]
        for a, b in PAIRS:
            ta = ex[a]["topK"][:K_jacc]
            tb = ex[b]["topK"][:K_jacc]
            jacc[f"{a},{b}"].append(jaccard(ta, tb))
        if has_spear_full:
            sf = e["spearman_full"]
            for a, b in PAIRS:
                spear_full[f"{a},{b}"].append(sf[f"{a},{b}"])

    jacc_mean = {k: float(np.mean(v)) for k, v in jacc.items()}
    spear_mean = (
        {k: float(np.mean(v)) for k, v in spear_full.items()}
        if has_spear_full else
        {k: None for k in spear_full}
    )

    # ---- RQ3.3 Lift attribution ----
    fusion_key = list(entries[0]["fusion"].keys())[0]
    cats = Counter()
    recovered = 0
    fusion_hit = 0
    sas_hit = 0
    icf_hit = 0
    sem_hit = 0

    for e in entries:
        ex = e["experts"]
        fu = e["fusion"][fusion_key]
        sas_h = hit_at_k(ex["SASRec"]["target_rank"], K_lift)
        icf_h = hit_at_k(ex["ItemCF"]["target_rank"], K_lift)
        sem_h = hit_at_k(ex["Semantic"]["target_rank"], K_lift)
        fu_h = hit_at_k(fu["target_rank"], K_lift)

        sas_hit += int(sas_h)
        icf_hit += int(icf_h)
        sem_hit += int(sem_h)
        fusion_hit += int(fu_h)

        # "Recovered" = fusion hits, SASRec misses
        if fu_h and not sas_h:
            recovered += 1
            if sem_h and icf_h:
                cats["both_sem_and_cf_hit"] += 1
            elif sem_h:
                cats["semantic_only_hit"] += 1
            elif icf_h:
                cats["itemcf_only_hit"] += 1
            else:
                cats["fusion_promotion (no expert top10)"] += 1

    attribution = {
        "n_users": n,
        "hit10": {
            "SASRec": sas_hit / n,
            "ItemCF": icf_hit / n,
            "Semantic": sem_hit / n,
            "Fusion": fusion_hit / n,
        },
        "n_recovered (Fusion@10 hit, SASRec@10 miss)": recovered,
        "recovered_pct": recovered / n if n else 0.0,
        "attribution_breakdown": {
            k: {"count": v, "pct_of_recovered": v / recovered if recovered else 0.0}
            for k, v in cats.items()
        },
    }

    return {
        "n_users": n,
        "K_jaccard": K_jacc,
        "K_lift": K_lift,
        "fusion_key": fusion_key,
        "rq3_1_jaccard_mean": jacc_mean,
        "rq3_2_spearman_mean": spear_mean,
        "rq3_3_lift_attribution": attribution,
        "rq3_4_hit_set_decomposition_pct": hit_set_decomposition(entries, K_lift),
    }


def cohort_split(entries, n_buckets=3):
    """Split entries into history-length terciles. Returns dict cohort -> entries."""
    if not entries:
        return {}
    lens = sorted(e["history_length"] for e in entries)
    n = len(lens)
    t1 = lens[n // 3]
    t2 = lens[2 * n // 3]
    buckets = {"hard": [], "medium": [], "easy": []}
    for e in entries:
        L = e["history_length"]
        if L <= t1:
            buckets["hard"].append(e)
        elif L <= t2:
            buckets["medium"].append(e)
        else:
            buckets["easy"].append(e)
    return buckets, (t1, t2)


def print_report(name, report, indent=""):
    print(f"\n{indent}=== {name} (n={report['n_users']}) ===")
    print(f"{indent}RQ3.1 Jaccard@{report['K_jaccard']} (expert pairs)")
    for pair, v in report["rq3_1_jaccard_mean"].items():
        print(f"{indent}  {pair:<20} {v:.4f}")
    print(f"{indent}RQ3.2 Spearman ρ (full item-vector, per-user mean)")
    for pair, v in report["rq3_2_spearman_mean"].items():
        if v is None:
            print(f"{indent}  {pair:<20} (not in JSONL — rerun dump with current script)")
        else:
            print(f"{indent}  {pair:<20} {v:+.4f}")
    print(f"{indent}RQ3.3 Lift attribution at K={report['K_lift']}")
    h = report["rq3_3_lift_attribution"]["hit10"]
    print(f"{indent}  hit@10  SASRec={h['SASRec']:.4f}  ItemCF={h['ItemCF']:.4f}  "
          f"Semantic={h['Semantic']:.4f}  Fusion={h['Fusion']:.4f}")
    att = report["rq3_3_lift_attribution"]
    print(f"{indent}  recovered_pct (Fusion@10 hit, SAS@10 miss): {att['recovered_pct']*100:.2f}%  "
          f"(n={att['n_recovered (Fusion@10 hit, SASRec@10 miss)']})")
    for cat, d in att["attribution_breakdown"].items():
        print(f"{indent}    {cat:<40} {d['count']:>6}  {d['pct_of_recovered']*100:.1f}% of recovered")
    print(f"{indent}RQ3.4 exact singleton hit sets at K={report['K_lift']} (%)")
    hset = report["rq3_4_hit_set_decomposition_pct"]
    print(f"{indent}  " + "  ".join(f"{key}={value:.2f}" for key, value in hset.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="per_user_metrics_{dataset}.jsonl")
    p.add_argument("--K-jacc", type=int, default=10)
    p.add_argument("--K-lift", type=int, default=10)
    p.add_argument("--cohort", action="store_true",
                   help="Also report per-cohort breakdown by history-length terciles")
    p.add_argument("--out-json", default=None, help="Write report to JSON file")
    args = p.parse_args()

    entries = list(iter_jsonl(args.input))
    print(f"[loaded] {args.input} ({len(entries)} users)")

    report_all = analyze(entries, K_jacc=args.K_jacc, K_lift=args.K_lift)
    print_report("ALL", report_all)

    out = {"all": report_all}

    if args.cohort:
        buckets, (t1, t2) = cohort_split(entries)
        print(f"\n[cohort] history-length terciles: hard ≤ {t1}, medium ≤ {t2}, easy > {t2}")
        out["cohort_thresholds"] = {"t1": t1, "t2": t2}
        for cohort_name in ("hard", "medium", "easy"):
            cohort_entries = buckets[cohort_name]
            rep = analyze(cohort_entries, K_jacc=args.K_jacc, K_lift=args.K_lift)
            if rep is not None:
                print_report(cohort_name, rep, indent="  ")
                out[cohort_name] = rep

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
