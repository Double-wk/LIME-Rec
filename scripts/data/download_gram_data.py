"""Download Beauty / Toys / Sports from the GRAM (ACL 2025) repository.

GRAM uses the standard Amazon 2014 (McAuley UCSD) 5-core preprocessing that
the entire SASRec / BERT4Rec / S3-Rec / TIGER / GRAM benchmark family is built
on. Using their preprocessed files is the only way to guarantee fully aligned
user / item universes for paper-level comparisons.

We pull two files per dataset:

    user_sequence.txt    -- one chronological sequence per line:
                            "<user_id> <item_1> <item_2> ... <item_N>"
    item_plain_text.txt  -- one item per line:
                            "<item_id> title: ...; brand: ...; categories: ...; ..."

The user sequence is converted to our project's jsonl interaction format with
a synthetic monotonically-increasing timestamp (= position in the sequence)
so the existing data loader keeps working unchanged. Because GRAM data is
already 5-core filtered, our iterative kcore is a no-op on top of it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE = "https://raw.githubusercontent.com/skleee/GRAM/main/rec_datasets"

DATASETS = {
    "beauty": {"remote": "Beauty", "local_dir": "data/amazon/beauty"},
    "toys": {"remote": "Toys", "local_dir": "data/amazon/toys"},
    "sports": {"remote": "Sports", "local_dir": "data/amazon/sports"},
    "yelp": {"remote": "Yelp", "local_dir": "data/yelp"},
}


def fetch(url: str, dest: Path) -> bool:
    print(f"  curl {url} -> {dest}", flush=True)
    result = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", "600", "-o", str(dest), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: curl exit={result.returncode} stderr={result.stderr.strip()}", flush=True)
        return False
    size_mb = dest.stat().st_size / 1e6
    print(f"  done: {size_mb:.2f} MB", flush=True)
    return True


def convert_sequences(seq_path: Path, jsonl_path: Path) -> tuple[int, int, int]:
    users, items, n = set(), set(), 0
    with seq_path.open("r", encoding="utf-8") as fin, jsonl_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            user_id, item_seq = toks[0], toks[1:]
            users.add(user_id)
            for t, item_id in enumerate(item_seq, start=1):
                items.add(item_id)
                fout.write(json.dumps({
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": 1.0,
                    "timestamp": t,
                }, ensure_ascii=False) + "\n")
                n += 1
    return len(users), len(items), n


def parse_item_text_line(line: str) -> dict:
    """Parse `ITEM_ID title: ...; brand: ...; categories: ...; description: ...; ...`."""
    line = line.rstrip("\n")
    if not line:
        return {}
    item_id, _, rest = line.partition(" ")
    if not item_id:
        return {}
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_buf: list[str] = []
    for chunk in rest.split("; "):
        key, sep, val = chunk.partition(": ")
        if sep and key in {"title", "brand", "categories", "category",
                           "description", "price", "salesrank"}:
            if current_key is not None:
                fields[current_key] = "; ".join(current_buf).strip()
            current_key = key
            current_buf = [val]
        else:
            current_buf.append(chunk)
    if current_key is not None:
        fields[current_key] = "; ".join(current_buf).strip()
    return {"item_id": item_id, **fields}


def convert_meta(meta_path: Path, out_path: Path) -> int:
    n = 0
    with meta_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            parsed = parse_item_text_line(line)
            if not parsed.get("item_id"):
                continue
            title = parsed.get("title", "")
            brand = parsed.get("brand", "")
            category = parsed.get("categories") or parsed.get("category") or ""
            description = parsed.get("description", "")
            price = parsed.get("price", "")
            salesrank = parsed.get("salesrank", "")
            text_parts = [p for p in [title, brand, category, description, price, salesrank] if p and p != "na"]
            fout.write(json.dumps({
                "item_id": parsed["item_id"],
                "title": title,
                "brand": brand,
                "category": category,
                "description": " | ".join(text_parts) if text_parts else title,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    for name in targets:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")
            continue
        info = DATASETS[name]
        remote = info["remote"]
        out_dir = Path(info["local_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name.upper()} (GRAM/{remote}) ===", flush=True)

        seq_raw = out_dir / "user_sequence.txt"
        meta_raw = out_dir / "item_plain_text.txt"
        if not fetch(f"{BASE}/{remote}/user_sequence.txt", seq_raw):
            continue
        meta_ok = fetch(f"{BASE}/{remote}/item_plain_text.txt", meta_raw)

        train_jsonl = out_dir / "train.jsonl"
        n_users, n_items, n_inter = convert_sequences(seq_raw, train_jsonl)
        print(f"  sequences -> {train_jsonl}", flush=True)
        print(f"  users={n_users} items_seen={n_items} interactions={n_inter}", flush=True)

        meta_jsonl_path = out_dir / "meta.jsonl"
        if meta_ok:
            n_meta = convert_meta(meta_raw, meta_jsonl_path)
            print(f"  metadata -> {meta_jsonl_path} ({n_meta} items)", flush=True)

        config_name = name if name == "yelp" else f"amazon_{name}"
        config_path = Path(f"configs/{config_name}.json")
        config = {
            "name": config_name,
            "domain": name,
            "interactions_path": f"{info['local_dir']}/train.jsonl",
            "metadata_path": f"{info['local_dir']}/meta.jsonl",
            "min_user_interactions": 5,
            "min_item_interactions": 5,
            "split_strategy": "leave_two_out",
            "eval_k": [5, 10],
            "source": "GRAM rec_datasets",
        }
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
        print(f"  config -> {config_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
