"""Export TIGER (LETTER implementation) test rankings in audit format.

Reuses LETTER-TIGER's SeqRecDataset / TestCollator / candidate trie and the
trained checkpoint, but iterates deterministically (shuffle=False) and writes
per-user top-20 rankings mapped back to raw item ids, so the shared recovery
audit can compare TIGER and the recovery witness user-by-user.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

LETTER_DIR = Path(__file__).resolve().parents[2] / "external" / "LETTER" / "LETTER-TIGER"
sys.path.insert(0, str(LETTER_DIR))

from collator import TestCollator            # noqa: E402
from data import SeqRecDataset               # noqa: E402
from generation_trie import Trie             # noqa: E402
from utils import prefix_allowed_tokens_fn   # noqa: E402


class _Args:
    dataset: str
    data_path: str
    index_file: str = ".index.json"
    max_his_len: int = 20
    his_sep: str = ", "
    add_prefix: bool = False
    only_train_response: bool = False
    train_prompt_sample_num: str = "1"
    train_data_sample_num: str = "-1"
    valid_prompt_id: int = 0
    sample_valid: bool = True
    valid_prompt_sample_num: int = 2
    tasks: str = "seqrec"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)          # e.g. Beauty
    parser.add_argument("--data-path", default=str(LETTER_DIR.parent / "data_lime"))
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--mapping", required=True)          # mapping.json from prepare_tiger_data
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=50)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=0,
                        help="Debug: limit to the first N test users.")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    config = T5Config.from_pretrained(args.ckpt_path)
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path).to(device)
    model.eval()

    dargs = _Args()
    dargs.dataset = args.dataset
    dargs.data_path = args.data_path
    test_data = SeqRecDataset(dargs, mode="test")
    if args.max_users:
        test_data.inter_data = test_data.inter_data[: args.max_users]
    collator = TestCollator(dargs, tokenizer)
    all_items = test_data.get_all_items()
    candidate_trie = Trie([[0] + tokenizer.encode(candidate) for candidate in all_items])
    prefix_fn = prefix_allowed_tokens_fn(candidate_trie)
    loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                        shuffle=False, num_workers=2, pin_memory=True)

    mapping = json.loads(Path(args.mapping).read_text())
    item_raw = mapping["item_int_to_raw"]
    user_raw = mapping["user_int_to_raw"]
    user_order = [user_raw[int(u)] for u in test_data.inters.keys()]
    # all_items entries are joined semantic-id strings; map string -> raw item id
    indices = json.loads((Path(args.data_path) / args.dataset /
                          f"{args.dataset}{dargs.index_file}").read_text())
    indexstr_to_raw = {"".join(toks): item_raw[int(k)] for k, toks in indices.items()}

    rows = []
    cursor = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            output = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=10, prefix_allowed_tokens_fn=prefix_fn,
                num_beams=args.num_beams, num_return_sequences=args.num_beams,
                output_scores=True, return_dict_in_generate=True, early_stopping=True,
            )
            seqs = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
            scores = output["sequences_scores"].detach().cpu().tolist()
            batch = len(targets)
            for b in range(batch):
                cand = seqs[b * args.num_beams:(b + 1) * args.num_beams]
                sc = scores[b * args.num_beams:(b + 1) * args.num_beams]
                pairs = sorted(zip(cand, sc), key=lambda x: -x[1])
                seen, ranked = set(), []
                for text, _ in pairs:
                    compact = text.strip().replace(" ", "")
                    raw = indexstr_to_raw.get(compact)
                    if raw is None or raw in seen:
                        continue
                    seen.add(raw)
                    ranked.append(raw)
                    if len(ranked) >= args.top_k:
                        break
                rows.append((user_order[cursor + b], targets[b], ranked))
            cursor += batch
            if cursor % 2000 < args.test_batch_size:
                print(f"[tiger-test] {cursor}/{len(test_data)}", flush=True)
    assert cursor == len(test_data), f"row count mismatch: {cursor} vs {len(test_data)}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for user_id, target_indexstr, ranked in rows:
            target_raw = indexstr_to_raw.get(target_indexstr.strip().replace(" ", ""), "")
            handle.write(json.dumps({
                "user_id": user_id, "target_item_id": target_raw, "ranking": ranked,
                "seed": args.seed, "dataset": f"amazon_{args.dataset.lower()}",
                "model": "TIGER-letter-bge"}) + "\n")
    print(f"[tiger-test] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
