"""Process MovieLens-1M into our format.

Format:
  ratings.dat:  UserID::MovieID::Rating::Timestamp
  movies.dat:   MovieID::Title (Year)::Genres (|-separated)

Output:
  data/ml1m/train.jsonl  — (user_id, item_id, timestamp) lines
  data/ml1m/meta.jsonl   — (item_id, title, brand="", category, description) lines
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def iterative_5core(records, min_inter=5, max_passes=20):
    print(f"[5-core] start: {len(records):,}", flush=True)
    for it in range(max_passes):
        u_cnt = Counter(r[0] for r in records)
        i_cnt = Counter(r[1] for r in records)
        before = len(records)
        records = [r for r in records
                   if u_cnt[r[0]] >= min_inter and i_cnt[r[1]] >= min_inter]
        after = len(records)
        n_u = len({r[0] for r in records})
        n_i = len({r[1] for r in records})
        print(f"  pass {it+1}: {before:,} → {after:,}  users={n_u:,}  items={n_i:,}",
              flush=True)
        if before == after:
            break
    return records


def main():
    # 1. Load ratings
    records = []
    with open("data/ml1m/ratings.dat", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            u, i, r, t = parts
            records.append((u, i, int(t)))
    print(f"[parse] {len(records):,} ratings", flush=True)

    # 2. 5-core filter
    records = iterative_5core(records)

    # 3. Write train.jsonl (sorted by timestamp per user)
    by_user = defaultdict(list)
    for u, i, t in records:
        by_user[u].append((t, i))
    n_users = 0
    n_inter = 0
    items_kept = set()
    with open("data/ml1m/train.jsonl", "w") as fp:
        for u, lst in by_user.items():
            lst.sort()
            for t, i in lst:
                rec = {"user_id": u, "item_id": i, "timestamp": t}
                fp.write(json.dumps(rec) + "\n")
                items_kept.add(i)
                n_inter += 1
            n_users += 1
    print(f"[train.jsonl] users={n_users:,} items={len(items_kept):,} "
          f"inter={n_inter:,}", flush=True)

    # 4. Load movie metadata, write meta.jsonl
    meta_by_id = {}
    with open("data/ml1m/movies.dat", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 3:
                continue
            iid, title, genres = parts
            meta_by_id[iid] = (title, genres)

    n_with = 0
    n_missing = 0
    with open("data/ml1m/meta.jsonl", "w") as fp:
        for iid in sorted(items_kept):
            if iid in meta_by_id:
                title, genres = meta_by_id[iid]
                n_with += 1
            else:
                title, genres = iid, ""
                n_missing += 1
            # genres: "Animation|Children's|Comedy" → "animation, children's, comedy"
            cat = genres.lower().replace("|", ", ")
            desc_parts = [title.lower()]
            if cat:
                desc_parts.append(cat)
            description = ". ".join(desc_parts)
            rec = {
                "item_id": iid,
                "title": title.lower(),
                "brand": "",  # ml-1m has no brand/director field
                "category": cat,
                "description": description,
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[meta.jsonl] with={n_with:,} missing={n_missing:,}", flush=True)


if __name__ == "__main__":
    main()
