"""Export LIGER (Meta, TMLR 2025) test rankings in audit format.

Reuses LIGER's load_data / TIGER model / generation loop (mirroring
src/evaluation.py::evaluate) plus, for the unified mode, the generate-then-
dense scoring of src/evaluation.py::generate_then_dense, and writes per-user
top-20 rankings mapped back to raw item ids so the shared recovery audit can
compare LIGER and the recovery witness user-by-user. All test users are
iterated in canonical order, including users whose test target is unseen in
training (LIGER's own cold-start users, which the published protocol scores
separately).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import T5Config

PROJ = Path(__file__).resolve().parents[2]
LIGER = PROJ / "external" / "LIGER"
sys.path.insert(0, str(LIGER))

from ID_generation.utils import process_data_split  # noqa: E402
from src.load_data import load_data  # noqa: E402
from src.tiger import TIGER  # noqa: E402
from src.evaluation import get_target_embed, model_forward  # noqa: E402
from utils import CustomDataset  # noqa: E402

NAME_MAP = {
    "beauty": "Beauty",
    "toys": "Toys_and_Games",
    "sports": "Sports_and_Outdoors",
    "yelp": "Yelp",
}
FEATURES = "title_price_brand_categories"


def build_config(args, name: str) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(LIGER / "configs"), version_base=None):
        cfg = compose(
            config_name="main",
            overrides=[
                "dataset=amazon",
                f"method={args.method}",
                "logging=wandb",
                f"dataset.name={name}",
                "dataset.content_model=bge-base",
                f"seed={args.seed}",
                f"device_id={args.device_id}",
                "logging.mode=disabled",
                f"test_method={args.mode}",
                f"experiment_id=lime_{args.method}_{name}_seed{args.seed}",
            ],
        )
    return OmegaConf.to_container(cfg, resolve=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(NAME_MAP))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="setting", choices=["base", "setting"])
    parser.add_argument("--mode", default=None, choices=["tiger", "liger"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=100,
                        help="LIGER's own evaluation uses beam=RETRIEVE_KEY max=100")
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--max-users", type=int, default=0,
                        help="Debug: export only the first N test users per split.")
    args = parser.parse_args()
    args.out = str(Path(args.out).resolve())  # resolve before chdir into LIGER
    if args.mode is None:
        args.mode = "tiger" if args.method == "base" else "liger"
    name = NAME_MAP[args.dataset]

    device = f"cuda:{args.device_id}" if args.device_id >= 0 else "cpu"
    os.chdir(LIGER)  # LIGER's set_dir and writers use repo-relative paths
    config = build_config(args, name)
    method_config = {
        **config["method"],
        **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]},
    }

    processed = LIGER / "ID_generation" / "preprocessing" / "processed"
    data_file = str(processed / f"{name}.txt")
    id2meta_file = str(processed / f"{name}_{FEATURES}_amazon_id2meta.json")
    id_save_location = str(
        LIGER / "ID_generation" / "ID" / f"{name}_bge-base_{args.seed}.pkl")

    id_split, user_sequence = process_data_split(
        config, data_file, id2meta_file, is_steam=False)
    item_embedding = torch.load(
        processed / f"{name}_bge-base_embeddings.pt", weights_only=False).to(device)

    train_config = {
        **config["dataset"],
        **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]},
    }
    codebook_size = train_config["RQ-VAE"]["code_book_size"]
    max_items_per_seq = train_config["max_items_per_seq"]
    (
        _training_data, _val_data, test_data, _unseen_val_data, unseen_test_data,
        _seen_sids, _val_unseen_sids, test_unseen_sids, max_last_semantic_ids,
        n_semantic_codebook, n_codebook, item2sid,
    ) = load_data(
        id_save_location, user_sequence,
        id_split["unseen_val"], id_split["unseen_test"], id_split["seen"],
        item_embedding, method_config,
        max_length=train_config["TIGER"]["n_positions"],
        codebook_size=codebook_size,
        max_items_per_seq=max_items_per_seq,
    )
    item2sid_tensor = torch.from_numpy(item2sid).to(device)
    test_unseen_sids_t = torch.from_numpy(test_unseen_sids).to(device)

    this_vocab_size = (codebook_size * n_semantic_codebook
                       + max(max_last_semantic_ids, codebook_size) + 2)
    t5 = train_config["TIGER"]["T5"]
    model_config = T5Config(
        num_layers=t5["encoder_layers"],
        num_decoder_layers=t5["decoder_layers"],
        d_model=t5["d_model"],
        d_ff=t5["d_ff"],
        num_heads=t5["num_heads"],
        d_kv=t5["d_kv"],
        dropout_rate=t5["dropout_rate"],
        vocab_size=this_vocab_size,
        pad_token_id=0,
        eos_token_id=int(this_vocab_size - 1),
        decoder_start_token_id=0,
        feed_forward_proj=t5["feed_forward_proj"],
        n_positions=train_config["TIGER"]["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5["initializer_factor"],
    )
    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
    ).to(device)
    ckpt = (LIGER / "results" / args.mode / f"Amazon_{name}"
            / f"lime_{args.method}_{name}_seed{args.seed}_seed_{args.seed}"
            / "results" / "ckpt_best.pt")
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False), strict=True)
    model.eval()

    # LIGER's own loader splits test users into test_data (target seen in
    # training) and unseen_test_data (cold target); recompute the assignment to
    # restore the canonical user order.
    unseen_test = id_split["unseen_test"]
    split_of = ["u" if seq[-1] in unseen_test else "t" for seq in user_sequence]

    loaders = {
        "t": DataLoader(CustomDataset(test_data), batch_size=args.test_batch_size,
                        shuffle=False),
        "u": DataLoader(CustomDataset(unseen_test_data), batch_size=args.test_batch_size,
                        shuffle=False),
    }
    max_per_split = args.max_users or float("inf")
    mapping = json.loads((processed / f"mapping_{name}.json").read_text())
    item_raw = mapping["item_int_to_raw"]

    user_rankings: dict[str, list[list[str]]] = {"t": [], "u": []}
    for key in ("t", "u"):
        cursor = 0
        for batch in loaders[key]:
            if cursor >= max_per_split:
                break
            labels = batch["labels_sids"].to(device)
            batch_size = labels.shape[0]
            with torch.no_grad():
                _, input_kwargs = model_forward(
                    model, batch, device, n_codebook, method_config, skip_forward=True)
                gen_kwargs = {
                    "num_beams": args.num_beams,
                    "max_new_tokens": n_codebook,
                    "num_return_sequences": args.num_beams,
                    "use_cache": True,
                }
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model.generate(**input_kwargs, **gen_kwargs)
                predicted_embedding = model.predicted_embedding
            outputs = outputs[:, 1:1 + n_codebook].reshape(
                batch_size, args.num_beams, -1)
            if outputs.shape[-1] < n_codebook:
                pad = torch.zeros(
                    (batch_size, args.num_beams, n_codebook - outputs.shape[-1]),
                    device=outputs.device, dtype=outputs.dtype)
                outputs = torch.cat([outputs, pad], dim=-1)

            for b in range(batch_size):
                cands = outputs[b]
                matches = torch.all(
                    cands[:, None, :] == item2sid_tensor[None, :, :], dim=-1)
                hit_positions = torch.argmax(matches.int(), dim=1)
                hit = matches.gather(1, hit_positions[:, None]).squeeze(1)
                matched = [int(hit_positions[p]) for p in range(args.num_beams)
                           if bool(hit[p])]

                if args.mode == "liger" and predicted_embedding is not None:
                    pred_emb = predicted_embedding.reshape(
                        batch_size, args.num_beams, -1)[:, 0][b]
                    # dense re-scoring over the top-RETRIEVE_KEY generative
                    # candidates plus the cold (unseen-test) items
                    pool = matched[:20]
                    if len(test_unseen_sids_t):
                        for sid in test_unseen_sids_t:
                            m = torch.all(item2sid_tensor == sid[None, :], dim=1)
                            if bool(m.any()):
                                pool.append(int(torch.argmax(m.int())))
                    if not pool:
                        user_rankings[key].append([])
                        continue
                    pool_emb = item_embedding[torch.tensor(pool, device=device)]
                    _, logits = get_target_embed(
                        pred_emb[None, :], model, method_config, pool_emb[None, ...])
                    top = torch.topk(
                        logits.squeeze(0), k=min(args.top_k, logits.shape[1]),
                        largest=True)[1].tolist()
                    ranking = [item_raw[pool[i]] for i in top]
                else:
                    ranking, seen_items = [], set()
                    for row in matched:
                        raw = item_raw[row]
                        if raw in seen_items:
                            continue
                        seen_items.add(raw)
                        ranking.append(raw)
                        if len(ranking) >= args.top_k:
                            break
                user_rankings[key].append(ranking)
            cursor += batch_size
        if not args.max_users:
            assert cursor == len(user_rankings[key])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counters = {"t": 0, "u": 0}
    with out.open("w", encoding="utf-8") as handle:
        for i, key in enumerate(split_of):
            local = counters[key]
            if local >= len(user_rankings[key]):
                continue  # debug subset run: users beyond the cap are skipped
            counters[key] += 1
            handle.write(json.dumps({
                "user_id": mapping["user_int_to_raw"][i],
                "target_item_id": item_raw[user_sequence[i][-1] - 1],
                "ranking": user_rankings[key][local],
                "seed": args.seed,
                "dataset": f"amazon_{args.dataset}" if args.dataset != "yelp" else "yelp",
                "model": f"liger-{args.method}-bge",
            }) + "\n")
    n_written = sum(len(v) for v in user_rankings.values())
    print(f"[liger-test] wrote {n_written} rows -> {out}")


if __name__ == "__main__":
    main()
