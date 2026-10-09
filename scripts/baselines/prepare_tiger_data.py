"""Convert a configured LIME-Rec dataset into LETTER-TIGER data format.

Writes, under ``<out_root>/<Dataset>/``:
  ``<Dataset>.inter.json``  {user_int: [item_int, ...]} chronological,
                            train history + valid + test (leave-two-out source)
  ``<Dataset>.item.json``   {item_int: {"title": ..., "description": ...}}
  ``<Dataset>.emb-bge.npy`` frozen BGE item embeddings in item_int order
  ``mapping.json``          item_int/user_int <-> raw id maps for export

Integer ids follow the canonical sorted order of the configured dataset, so
the TIGER audit runs on exactly the same users, items, splits, and text fields
as the controlled GRAM audit. The RQ-VAE tokenizer consumes the same frozen
BGE prior as the recovery witness's semantic expert.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lime_rec.evaluation import load_configured_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--semantic-emb", required=True,
                        help="npz with item_ids + embeddings (BGE prior)")
    parser.add_argument("--out-root", required=True,
                        help="e.g. external/LETTER/data_lime")
    args = parser.parse_args()

    dataset = load_configured_dataset(args.config)

    # Raw title/description fields for LETTER-style item.json (text is only
    # carried for provenance; TIGER consumes semantic IDs, not text).
    with open(args.config, encoding="utf-8") as fp:
        meta_path = Path(json.load(fp)["metadata_path"])
    raw_meta = {}
    with open(meta_path, encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            asin = str(row.get("asin") or row.get("item_id") or "")
            if asin:
                raw_meta[asin] = {
                    "title": str(row.get("title") or "").strip(),
                    "description": str(row.get("description") or "").strip(),
                }

    item_ids = list(dataset.item_ids)          # sorted catalog order
    user_ids = list(dataset.user_ids)          # sorted user order
    item_int = {raw: i for i, raw in enumerate(item_ids)}
    user_int = {raw: i for i, raw in enumerate(user_ids)}

    inter = {}
    for user in user_ids:
        seq = list(dataset.history_by_user[user])
        valid = dataset.valid_by_user.get(user)
        test = dataset.test_by_user.get(user)
        if valid:
            seq.append(valid)
        if test:
            seq.append(test)
        inter[str(user_int[user])] = [item_int[i] for i in seq]

    item_json = {
        str(item_int[raw]): raw_meta.get(raw, {"title": "", "description": ""})
        for raw in item_ids
    }

    with np.load(args.semantic_emb, allow_pickle=True) as emb:
        emb_ids = [str(x) for x in emb["item_ids"].tolist()]
        emb_matrix = emb["embeddings"][:]  # materialize once; npz re-reads per access
    pos = {raw: i for i, raw in enumerate(emb_ids)}
    matrix = np.stack([emb_matrix[pos[raw]] for raw in item_ids]).astype(np.float32)

    out_dir = Path(args.out_root) / dataset.name.replace("amazon_", "").title()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset.name.replace("amazon_", "").title()
    (out_dir / f"{stem}.inter.json").write_text(json.dumps(inter))
    (out_dir / f"{stem}.item.json").write_text(json.dumps(item_json))
    np.save(out_dir / f"{stem}.emb-bge.npy", matrix)
    (out_dir / "mapping.json").write_text(json.dumps({
        "item_int_to_raw": item_ids,
        "user_int_to_raw": user_ids,
        "config": args.config,
        "semantic_emb": args.semantic_emb,
    }, indent=2))
    print(f"[tiger-data] {dataset.name}: users={len(user_ids)} items={len(item_ids)} "
          f"emb={matrix.shape} -> {out_dir}")


if __name__ == "__main__":
    main()
