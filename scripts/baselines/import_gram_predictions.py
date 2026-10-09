"""Convert official GRAM TSV predictions to raw-item controlled JSONL.

GRAM 官方 TSV 的两个格式怪癖（已在 skleee/GRAM 实测确认）：
1. 表头硬编码 `idx H@5 H@10 NDCG@5 NDCG@10 gold pred scores`，但每行实际写入
   全部 12 个指标值 → 列名与数据错位，必须按位置解析（gold/pred/scores 为
   倒数第 3/2/1 列）。
2. gold 与 pred 写入的是 tokenizer.batch_decode 后的脱标记文本（空格连接、
   无 ▁/|），与语义映射文件的键格式（|▁token|...）不一致 → 需要先用
   t5-small tokenizer 对映射键做同样的 decode 构造反解索引，再查表。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lime_rec.baselines.gram import read_semantic_mapping, read_user_sequences
from lime_rec.controlled_metrics import stable_unique_ranking
from lime_rec.evaluation import load_configured_dataset


def build_detok_index(mapping: dict, tokenizer_name: str) -> tuple[dict, int]:
    """detok(语义ID) -> raw item 反解索引；返回 (索引, 冲突数)。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    index: dict[str, str] = {}
    collisions = 0
    for key, raw in mapping.items():
        tokens = [t for t in key.split("|") if t]
        text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
        )
        if index.get(text, raw) != raw:
            collisions += 1
        index.setdefault(text, raw)
    return index, collisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gram-tsv", required=True)
    parser.add_argument("--gram-user-sequence", required=True)
    parser.add_argument("--semantic-mapping", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tokenizer", default="t5-small")
    args = parser.parse_args()
    dataset = load_configured_dataset(args.config)
    sequence_users = list(read_user_sequences(args.gram_user_sequence))
    mapping, collisions = read_semantic_mapping(args.semantic_mapping)
    mapped_catalog = set(mapping.values())
    catalog = set(dataset.item_ids)
    coverage = len(mapped_catalog & catalog) / max(len(catalog), 1)
    if mapped_catalog != catalog or collisions or len(mapping) != len(catalog):
        raise RuntimeError("GRAM semantic mapping does not bijectively cover the LIME catalog")
    detok_index, detok_collisions = build_detok_index(mapping, args.tokenizer)
    if detok_collisions:
        raise RuntimeError(f"detokenized semantic IDs are not unique: {detok_collisions} collisions")

    rows: list[dict] = []
    unknown = duplicates = predictions = gold_target_mismatches = 0
    seen_users: set[str] = set()
    duplicate_user_rows = 0
    with Path(args.gram_tsv).open(encoding="utf-8") as handle:
        handle.readline()  # 表头与行布局错位（见模块 docstring），仅作跳过
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue  # 文件末尾的 hit@/ndcg@ 汇总行（无 \t 分列）
            identifier = parts[0].strip()
            if not identifier or identifier.startswith(("hit@", "ndcg@")):
                continue
            if identifier in dataset.test_by_user:
                user_id = identifier
            elif identifier.isdigit() and int(identifier) < len(sequence_users):
                user_id = sequence_users[int(identifier)]
            else:
                raise RuntimeError(f"unknown GRAM prediction user identifier: {identifier}")
            if user_id in seen_users:
                # DDP DistributedSampler 为整除会补齐重复末尾样本；跳过重复用户行
                duplicate_user_rows += 1
                continue
            seen_users.add(user_id)
            gold_raw = detok_index.get(parts[-3].strip())
            if gold_raw != dataset.test_by_user[user_id]:
                gold_target_mismatches += 1
            generated = parts[-2].split("||")
            raw: list[str] = []
            for semantic_id in generated:
                predictions += 1
                item_id = detok_index.get(semantic_id.strip())
                if item_id is None:
                    unknown += 1
                else:
                    raw.append(item_id)
            raw, duplicate_count = stable_unique_ranking(raw)
            duplicates += duplicate_count
            rows.append({"user_id": user_id, "target_item_id": dataset.test_by_user[user_id],
                         "ranking": raw[:args.top_k], "seed": args.seed,
                         "dataset": dataset.name, "model": "GRAM"})
    if {row["user_id"] for row in rows} != set(dataset.user_ids):
        raise RuntimeError("GRAM TSV does not contain exactly one prediction row per evaluation user")
    if len(rows) != len(dataset.user_ids) or gold_target_mismatches:
        raise RuntimeError("GRAM TSV contains duplicate users or gold/test target mismatches")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    audit = {"mapping_coverage": coverage, "collision_count": collisions,
             "detok_collision_count": detok_collisions,
             "unknown_prediction_count": unknown,
             "unknown_prediction_rate": unknown / max(predictions, 1),
             "duplicate_prediction_count": duplicates,
             "duplicate_prediction_rate": duplicates / max(predictions, 1),
             "gold_target_mismatches": gold_target_mismatches,
             "duplicate_user_rows": duplicate_user_rows,
             "num_prediction_rows": len(rows)}
    audit_path = Path(args.audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
