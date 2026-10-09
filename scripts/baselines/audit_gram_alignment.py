"""Assert exact sequence/target/catalog alignment against official GRAM data."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lime_rec.baselines.gram import read_semantic_mapping, read_user_sequences, sha256_file
from lime_rec.evaluation import load_configured_dataset


def audit(config_path: str, gram_sequence_path: str, gram_root: str,
          semantic_mapping_path: str | None = None) -> dict:
    dataset = load_configured_dataset(config_path)
    gram = read_user_sequences(gram_sequence_path)
    lime_sequences = {u: dataset.history_by_user[u] + [dataset.valid_by_user[u], dataset.test_by_user[u]]
                      for u in dataset.user_ids}
    lime_users, gram_users = set(lime_sequences), set(gram)
    lime_items = {item for sequence in lime_sequences.values() for item in sequence}
    gram_items = {item for sequence in gram.values() for item in sequence}
    shared = lime_users & gram_users
    target_mismatches = sum(lime_sequences[u][-2:] != gram[u][-2:] for u in shared)
    sequence_mismatches = sum(lime_sequences[u] != gram[u] for u in shared)
    root = Path(gram_root)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            text=True, capture_output=True).stdout.strip()
    result = {"dataset": dataset.name, "num_users": dataset.num_users,
              "num_items": dataset.num_items,
              "num_interactions": sum(map(len, lime_sequences.values())),
              "missing_users": len(lime_users - gram_users), "extra_users": len(gram_users - lime_users),
              "missing_items": len(lime_items - gram_items), "extra_items": len(gram_items - lime_items),
              "target_mismatches": target_mismatches, "sequence_mismatches": sequence_mismatches,
              "gram_commit": commit,
              "sha256": {"lime_interactions": sha256_file(json.loads(Path(config_path).read_text())["interactions_path"]),
                         "gram_user_sequence": sha256_file(gram_sequence_path)}}
    if semantic_mapping_path:
        mapping, collisions = read_semantic_mapping(semantic_mapping_path)
        mapped_items = set(mapping.values())
        result["mapping_coverage"] = len(mapped_items & lime_items) / max(len(lime_items), 1)
        result["semantic_id_collision_count"] = collisions
        result["mapping_duplicate_raw_item_count"] = len(mapping) - len(mapped_items)
        result["mapping_missing_items"] = len(lime_items - mapped_items)
        result["mapping_extra_items"] = len(mapped_items - lime_items)
        result["sha256"]["gram_semantic_mapping"] = sha256_file(semantic_mapping_path)
    result["passed"] = all(result[key] == 0 for key in
                           ("missing_users", "extra_users", "missing_items", "extra_items",
                            "target_mismatches", "sequence_mismatches"))
    if semantic_mapping_path:
        result["passed"] = result["passed"] and all(result[key] == 0 for key in
            ("semantic_id_collision_count", "mapping_missing_items", "mapping_extra_items"))
        result["passed"] = result["passed"] and result["mapping_duplicate_raw_item_count"] == 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gram-user-sequence", required=True)
    parser.add_argument("--gram-root", default="external/GRAM")
    parser.add_argument("--semantic-mapping")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.config, args.gram_user_sequence, args.gram_root, args.semantic_mapping)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
