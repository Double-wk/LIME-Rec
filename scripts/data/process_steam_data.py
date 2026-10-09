"""Process raw Steam reviews + games metadata into our standard format.

Steam reviews JSON (Python dict-literal lines) → train.jsonl + meta.jsonl
- 5-core filter (iterative)
- Sort each user's interactions by date
- Output:
    data/steam/train.jsonl  — one (user_id, item_id, timestamp) per line
    data/steam/meta.jsonl   — one item metadata record per line

The raw file is ~1.3GB. We parse line by line to avoid memory blow-up.
"""
from __future__ import annotations

import ast
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REVIEWS_PATH = "data/steam/steam_reviews.json.gz"
GAMES_PATH = "data/steam/steam_games.json.gz"
TRAIN_OUT = "data/steam/train.jsonl"
META_OUT = "data/steam/meta.jsonl"


def parse_date(s: str) -> int | None:
    """YYYY-MM-DD → unix epoch. Returns None on failure."""
    if not s:
        return None
    try:
        return int(datetime.strptime(s.strip(), "%Y-%m-%d").timestamp())
    except Exception:
        return None


def iter_reviews():
    """Yield (user_id, item_id, ts) from raw reviews. Skip records without
    needed fields. Use username as user_id (canonical in LIGER/TIGER).
    """
    n_total = 0
    n_kept = 0
    with gzip.open(REVIEWS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                d = ast.literal_eval(line)
            except Exception:
                continue
            uid = d.get("user_id") or d.get("username")
            iid = d.get("product_id")
            ts = parse_date(d.get("date", ""))
            if not uid or not iid or ts is None:
                continue
            n_kept += 1
            yield str(uid), str(iid), ts
            if n_total % 500000 == 0:
                print(f"  parsed {n_total:,} lines, kept {n_kept:,}",
                      flush=True)
    print(f"[parse] total={n_total:,} kept={n_kept:,}", flush=True)


def iterative_5core(records, min_inter: int = 5, max_passes: int = 20):
    """Iterative 5-core filter on (user, item, ts) triples.

    Returns the filtered list.
    """
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


def load_games_meta():
    """Build item_id → metadata dict from steam_games.json.gz."""
    meta = {}
    with gzip.open(GAMES_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = ast.literal_eval(line)
            except Exception:
                continue
            iid = d.get("id")
            if not iid:
                continue
            meta[str(iid)] = d
    print(f"[meta] loaded {len(meta):,} games", flush=True)
    return meta


def write_meta(item_ids, games_meta):
    """For each item, emit a JSONL line with title/genre/tags/developer text."""
    n_with_meta = 0
    n_missing = 0
    with open(META_OUT, "w", encoding="utf-8") as fp:
        for iid in item_ids:
            m = games_meta.get(iid, {})
            if m:
                n_with_meta += 1
            else:
                n_missing += 1
            title = m.get("title") or m.get("app_name") or ""
            genres = m.get("genres") or []
            tags = m.get("tags") or []
            developer = m.get("developer") or ""
            specs = m.get("specs") or []
            # Build category string: genres + tags
            cat_parts = []
            if genres:
                cat_parts.extend(genres if isinstance(genres, list) else [str(genres)])
            if tags:
                cat_parts.extend(tags if isinstance(tags, list) else [str(tags)])
            category = ", ".join(cat_parts[:10])  # cap to avoid bloat
            desc_parts = []
            if title:
                desc_parts.append(title)
            if developer:
                desc_parts.append(f"by {developer}")
            if specs:
                desc_parts.append("specs: " + ", ".join(specs[:5]))
            description = ". ".join(desc_parts)
            rec = {
                "item_id": iid,
                "title": title.lower(),
                "brand": developer.lower(),
                "category": category.lower(),
                "description": description.lower(),
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[meta] wrote {len(item_ids):,} items: with_meta={n_with_meta:,} missing={n_missing:,}",
          flush=True)


def main():
    Path("data/steam").mkdir(parents=True, exist_ok=True)

    # 1. Parse all reviews
    records = list(iter_reviews())
    print(f"[parse] {len(records):,} valid records", flush=True)

    # Dedupe (some users may review same game twice)
    seen = set()
    dedup = []
    for u, i, t in records:
        key = (u, i)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((u, i, t))
    print(f"[dedupe] {len(records):,} → {len(dedup):,}", flush=True)
    records = dedup

    # 2. 5-core filter
    records = iterative_5core(records, min_inter=5)

    # 3. Write train.jsonl (sorted by ts within each user)
    by_user = defaultdict(list)
    for u, i, t in records:
        by_user[u].append((t, i))
    n_users = 0
    n_inter = 0
    items_kept = set()
    with open(TRAIN_OUT, "w", encoding="utf-8") as fp:
        for u, lst in by_user.items():
            lst.sort()
            for t, i in lst:
                rec = {"user_id": u, "item_id": i, "timestamp": t}
                fp.write(json.dumps(rec) + "\n")
                items_kept.add(i)
                n_inter += 1
            n_users += 1
    print(f"[train.jsonl] users={n_users:,} items={len(items_kept):,} "
          f"interactions={n_inter:,}", flush=True)

    # 4. Write meta.jsonl
    print(f"[meta] loading games metadata...", flush=True)
    games_meta = load_games_meta()
    write_meta(sorted(items_kept), games_meta)

    print(f"\n[done] outputs:")
    print(f"  {TRAIN_OUT}")
    print(f"  {META_OUT}")


if __name__ == "__main__":
    main()
