"""Process Amazon CDs_and_Vinyl 5-core into our train.jsonl + meta.jsonl format.

Raw inputs (already downloaded under data/amazon/music/raw/):
  reviews_CDs_and_Vinyl_5.json.gz  -- one JSON object per line:
    {"reviewerID", "asin", "unixReviewTime", "overall", "reviewText", ...}
  meta_CDs_and_Vinyl.json.gz       -- one Python dict-literal per line:
    {'asin': ..., 'title': ..., 'brand': ..., 'categories': [[...]], 'description': ...}

Outputs match the existing B/T/S format under data/amazon/{beauty,toys,sports}/:
  data/amazon/music/train.jsonl    -- {"user_id","item_id","rating","timestamp"}
  data/amazon/music/meta.jsonl     -- {"item_id","title","brand","category","description"}

All text is lowercased to match the B/T/S corpus convention.
"""
from __future__ import annotations

import ast
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


REVIEWS = "data/amazon/music/raw/reviews_CDs_and_Vinyl_5.json.gz"
META = "data/amazon/music/raw/meta_CDs_and_Vinyl.json.gz"
TRAIN_OUT = "data/amazon/music/train.jsonl"
META_OUT = "data/amazon/music/meta.jsonl"


def iter_reviews():
    n_total = 0
    n_kept = 0
    with gzip.open(REVIEWS, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                d = json.loads(line)
            except Exception:
                try:
                    d = ast.literal_eval(line)
                except Exception:
                    continue
            uid = d.get("reviewerID")
            iid = d.get("asin")
            ts = d.get("unixReviewTime")
            rating = float(d.get("overall", 0.0) or 0.0)
            if not uid or not iid or ts is None:
                continue
            n_kept += 1
            yield str(uid), str(iid), int(ts), rating
            if n_total % 200000 == 0:
                print(f"  parsed {n_total:,} lines, kept {n_kept:,}", flush=True)
    print(f"[parse] reviews: total={n_total:,} kept={n_kept:,}", flush=True)


def iter_meta():
    n_total = 0
    n_kept = 0
    with gzip.open(META, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                d = ast.literal_eval(line)
            except Exception:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
            iid = d.get("asin")
            if not iid:
                continue
            n_kept += 1
            yield str(iid), d
            if n_total % 100000 == 0:
                print(f"  meta parsed {n_total:,}", flush=True)
    print(f"[parse] meta: total={n_total:,} kept={n_kept:,}", flush=True)


def iterative_5core(records, min_inter: int = 5, max_passes: int = 20):
    print(f"[5-core] start: {len(records):,} records", flush=True)
    for it in range(max_passes):
        user_count = Counter(r[0] for r in records)
        item_count = Counter(r[1] for r in records)
        before = len(records)
        records = [r for r in records
                   if user_count[r[0]] >= min_inter and item_count[r[1]] >= min_inter]
        after = len(records)
        n_users = len({r[0] for r in records})
        n_items = len({r[1] for r in records})
        print(f"  pass {it+1}: {before:,} → {after:,}  users={n_users:,}  items={n_items:,}",
              flush=True)
        if before == after:
            break
    return records


def build_item_text(m: dict) -> tuple[str, str, str, str]:
    """Return (title, brand, category, description), each lowercased."""
    title = (m.get("title") or "").strip().lower()
    brand = (m.get("brand") or "").strip().lower()
    cats = m.get("categories") or []
    cat_words = []
    for c in cats:
        if isinstance(c, list):
            cat_words.extend([str(x).strip() for x in c if x])
        elif c:
            cat_words.append(str(c).strip())
    category = ", ".join([c.lower() for c in cat_words[:8]])
    desc = (m.get("description") or "").strip().lower()
    desc = " ".join(desc.split())[:2000]  # cap length
    return title, brand, category, desc


def main():
    Path("data/amazon/music").mkdir(parents=True, exist_ok=True)

    # 1. Parse + dedupe reviews
    records = list(iter_reviews())
    print(f"[parse] {len(records):,} valid review records", flush=True)

    seen = {}
    for u, i, t, r in records:
        key = (u, i)
        if key in seen and seen[key][0] <= t:
            continue
        seen[key] = (t, r)
    records = [(u, i, t, r) for (u, i), (t, r) in seen.items()]
    print(f"[dedupe] {len(records):,} unique (user, item) pairs", flush=True)

    # 2. 5-core filter (data is _5 but be defensive)
    records = iterative_5core(records, min_inter=5)

    # 3. Write train.jsonl
    by_user = defaultdict(list)
    for u, i, t, r in records:
        by_user[u].append((t, i, r))
    n_users = 0
    n_inter = 0
    items_kept = set()
    with open(TRAIN_OUT, "w", encoding="utf-8") as fp:
        for u, lst in by_user.items():
            lst.sort()
            for t, i, r in lst:
                fp.write(json.dumps({
                    "user_id": u, "item_id": i, "rating": r, "timestamp": t
                }) + "\n")
                items_kept.add(i)
                n_inter += 1
            n_users += 1
    print(f"[train.jsonl] users={n_users:,} items={len(items_kept):,} interactions={n_inter:,}",
          flush=True)

    # 4. Write meta.jsonl
    print(f"[meta] loading metadata...", flush=True)
    meta_by_id = {iid: m for iid, m in iter_meta()}
    n_with = 0
    n_missing = 0
    with open(META_OUT, "w", encoding="utf-8") as fp:
        for iid in sorted(items_kept):
            m = meta_by_id.get(iid, {})
            if m:
                n_with += 1
            else:
                n_missing += 1
            title, brand, category, desc = build_item_text(m)
            fp.write(json.dumps({
                "item_id": iid,
                "title": title,
                "brand": brand,
                "category": category,
                "description": desc,
            }, ensure_ascii=False) + "\n")
    print(f"[meta.jsonl] wrote {len(items_kept):,} items: with_meta={n_with:,} missing={n_missing:,}",
          flush=True)

    print(f"\n[done] outputs:")
    print(f"  {TRAIN_OUT}")
    print(f"  {META_OUT}")


if __name__ == "__main__":
    main()
