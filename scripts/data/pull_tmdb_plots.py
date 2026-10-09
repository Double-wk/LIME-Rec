"""Pull TMDB plot summaries for ML-1M movies.

Why: ML-1M's only item text is "title (year)" + 18 unique genre tokens. SBERT on
that is essentially noise. TMDB provides 200-500 char plot summaries; we test
whether richer text closes the gap on ML-1M's -4.1% fusion failure.

Usage:
  export TMDB_API_KEY=your_key_here  # https://www.themoviedb.org/settings/api
  python pull_tmdb_plots.py

Output:
  data/ml1m/tmdb_plots.jsonl   — {item_id, title, year, tmdb_id, overview, tagline}
  data/ml1m/tmdb_misses.jsonl  — items where no TMDB match found

Resumable: re-running skips items already in tmdb_plots.jsonl.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

API_BASE = "https://api.themoviedb.org/3"
META_PATH = Path("data/ml1m/meta.jsonl")
OUT_PATH = Path("data/ml1m/tmdb_plots.jsonl")
MISS_PATH = Path("data/ml1m/tmdb_misses.jsonl")

TITLE_YEAR_RE = re.compile(r"^(.*)\s*\((\d{4})\)\s*$")


def parse_title_year(s: str) -> tuple[str, int | None]:
    m = TITLE_YEAR_RE.match(s.strip())
    if not m:
        return s.strip(), None
    title = m.group(1).strip()
    year = int(m.group(2))
    # ml-1m puts ", The" / ", A" suffix: "matrix, the (1999)" → "the matrix"
    for suf in (", the", ", a", ", an"):
        if title.lower().endswith(suf):
            title = title[: -len(suf)].strip()
            title = suf.strip(", ").capitalize() + " " + title
            break
    return title, year


def tmdb_search(title: str, year: int | None, api_key: str) -> dict | None:
    params = {"api_key": api_key, "query": title, "include_adult": "false"}
    if year:
        params["year"] = year
    for attempt in range(3):
        try:
            r = requests.get(f"{API_BASE}/search/movie", params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            results = r.json().get("results", [])
            if not results and year:
                # retry without year constraint (ml-1m years are sometimes off-by-one)
                params2 = {k: v for k, v in params.items() if k != "year"}
                r2 = requests.get(f"{API_BASE}/search/movie", params=params2, timeout=10)
                r2.raise_for_status()
                results = r2.json().get("results", [])
            return results[0] if results else None
        except requests.RequestException as e:
            if attempt == 2:
                print(f"  [error] {title}: {e}", flush=True)
                return None
            time.sleep(1 + attempt)
    return None


def tmdb_details(tmdb_id: int, api_key: str) -> dict | None:
    """Detail call to also get tagline (some items have null overview but useful tagline)."""
    params = {"api_key": api_key}
    try:
        r = requests.get(f"{API_BASE}/movie/{tmdb_id}", params=params, timeout=10)
        if r.status_code == 429:
            time.sleep(1)
            r = requests.get(f"{API_BASE}/movie/{tmdb_id}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def main():
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        print("ERROR: set TMDB_API_KEY env var.", file=sys.stderr)
        print("  Register free at https://www.themoviedb.org/settings/api", file=sys.stderr)
        sys.exit(1)

    items = []
    with open(META_PATH) as f:
        for line in f:
            items.append(json.loads(line))
    print(f"[load] {len(items):,} ml-1m items", flush=True)

    # Resume support
    done_ids = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                done_ids.add(json.loads(line)["item_id"])
        print(f"[resume] {len(done_ids):,} already fetched", flush=True)

    n_hit = len(done_ids)
    n_miss = 0
    t0 = time.time()

    out_fp = open(OUT_PATH, "a")
    miss_fp = open(MISS_PATH, "a")

    for idx, it in enumerate(items):
        iid = it["item_id"]
        if iid in done_ids:
            continue

        title, year = parse_title_year(it["title"])
        hit = tmdb_search(title, year, api_key)
        if not hit:
            n_miss += 1
            miss_fp.write(json.dumps({"item_id": iid, "title": it["title"]}) + "\n")
            miss_fp.flush()
            continue

        # /search returns overview too; only call details if overview empty
        overview = (hit.get("overview") or "").strip()
        tagline = ""
        if not overview:
            det = tmdb_details(hit["id"], api_key)
            if det:
                overview = (det.get("overview") or "").strip()
                tagline = (det.get("tagline") or "").strip()

        out_fp.write(json.dumps({
            "item_id": iid,
            "title": title,
            "year": year,
            "tmdb_id": hit["id"],
            "overview": overview,
            "tagline": tagline,
        }, ensure_ascii=False) + "\n")
        out_fp.flush()
        n_hit += 1

        if (idx + 1) % 100 == 0:
            rate = (idx + 1) / max(time.time() - t0, 1e-6)
            eta = (len(items) - idx - 1) / max(rate, 1e-6)
            print(f"  [{idx+1:,}/{len(items):,}] hit={n_hit:,} miss={n_miss:,} "
                  f"rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

        # polite throttle: ~10 req/s
        time.sleep(0.1)

    out_fp.close()
    miss_fp.close()
    print(f"[done] hit={n_hit:,} miss={n_miss:,} of {len(items):,}", flush=True)


if __name__ == "__main__":
    main()
