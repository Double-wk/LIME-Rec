"""Convert a configured LIME-Rec dataset into LIGER (Meta, TMLR 2025) data format.

Writes, under ``<liger-root>/ID_generation/preprocessing/processed/``:
  ``<Name>.txt``                        "user_int item_int ..." chronological,
                                        train history + valid + test (leave-two-out)
  ``<Name>_<content_model>_embeddings.pt``
                                        StandardScaler-normalized frozen BGE item
                                        embeddings in row order item_int - 1, the
                                        exact tensor LIGER's process_embeddings
                                        caches (sentence-t5-xxl is bypassed, so the
                                        RQ-VAE tokenizer consumes the same frozen
                                        BGE prior as the recovery witness)
  ``<Name>_<features>_<prompt>_id2meta.json``
                                        zero-padded item_int keys (lexicographic ==
                                        numeric order, required by LIGER's
                                        item_to_ind assertion)
  ``<Name>_item2attributes.json``       empty dict (attributes unused here)
  ``mapping.json``                      item_int/user_int <-> raw id maps

Integer ids follow the canonical sorted order of the configured dataset, so the
LIGER audit runs on exactly the same users, items, splits, and text fields as
the controlled GRAM audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from lime_rec.evaluation import load_configured_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--semantic-emb", required=True,
                        help="npz with item_ids + embeddings (frozen BGE prior)")
    parser.add_argument("--liger-root", default="external/LIGER")
    parser.add_argument("--content-model", default="bge-base")
    parser.add_argument("--features", default="title_price_brand_categories")
    parser.add_argument("--prompt-format", default="amazon")
    args = parser.parse_args()

    dataset = load_configured_dataset(args.config)
    name = dataset.name.replace("amazon_", "").title()
    if name == "Toys":
        name = "Toys_and_Games"
    elif name == "Sports":
        name = "Sports_and_Outdoors"

    item_ids = list(dataset.item_ids)
    user_ids = list(dataset.user_ids)
    item_int = {raw: i + 1 for i, raw in enumerate(item_ids)}   # LIGER ids start at 1
    user_int = {raw: i + 1 for i, raw in enumerate(user_ids)}

    width = max(6, len(str(len(item_ids) + 1)))

    processed = Path(args.liger_root) / "ID_generation" / "preprocessing" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    data_file = processed / f"{name}.txt"
    with data_file.open("w", encoding="utf-8") as fp:
        for user in user_ids:
            seq = list(dataset.history_by_user[user])
            valid = dataset.valid_by_user.get(user)
            test = dataset.test_by_user.get(user)
            if valid:
                seq.append(valid)
            if test:
                seq.append(test)
            fp.write(f"{user_int[user]} " + " ".join(str(item_int[i]) for i in seq) + "\n")

    id2meta = {}
    for raw in item_ids:
        key = str(item_int[raw]).zfill(width)
        id2meta[key] = f"{raw}"
    (processed / f"{name}_{args.features}_{args.prompt_format}_id2meta.json").write_text(
        json.dumps(id2meta))
    (processed / f"{name}_item2attributes.json").write_text(json.dumps({}))

    with np.load(args.semantic_emb, allow_pickle=True) as emb:
        emb_ids = [str(x) for x in emb["item_ids"].tolist()]
        emb_matrix = emb["embeddings"][:]
    pos = {raw: i for i, raw in enumerate(emb_ids)}
    matrix = np.stack([emb_matrix[pos[raw]] for raw in item_ids]).astype(np.float32)
    scaled = StandardScaler().fit_transform(matrix)   # LIGER's own embedding step
    torch.save(torch.tensor(scaled, dtype=torch.float32),
               processed / f"{name}_{args.content_model}_embeddings.pt")

    (processed / f"mapping_{name}.json").write_text(json.dumps({
        "item_int_to_raw": item_ids,
        "user_int_to_raw": user_ids,
        "config": args.config,
        "semantic_emb": args.semantic_emb,
        "content_model": args.content_model,
    }, indent=2))
    print(f"[liger-data] {name}: users={len(user_ids)} items={len(item_ids)} "
          f"emb={matrix.shape} -> {processed}")


if __name__ == "__main__":
    main()
