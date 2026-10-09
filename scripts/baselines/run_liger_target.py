"""Drive LIGER (Meta, TMLR 2025) as a third generative-retrieval audit target.

Mirrors external/LIGER/run.py's main() but consumes the canonical-split data
produced by scripts.baselines.prepare_liger_data (no raw-data preprocessing) and
the frozen BGE embeddings cached by the same converter, so the RQ-VAE tokenizer
and the dense head consume the same item-representation prior as the recovery
witness (matched-prior audit, same convention as the LETTER-TIGER audit).

Usage:
  python -m scripts.baselines.run_liger_target --dataset beauty --seed 0 \
      --method setting --mode liger --device-id 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
LIGER = PROJ / "external" / "LIGER"
sys.path.insert(0, str(LIGER))

NAME_MAP = {
    "beauty": "Beauty",
    "toys": "Toys_and_Games",
    "sports": "Sports_and_Outdoors",
    "yelp": "Yelp",
}

FEATURES = "title_price_brand_categories"


def build_config(args, name: str):
    from hydra import compose, initialize_config_dir

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
                *args.set,
            ],
        )
    return cfg  # DictConfig: LIGER's setup_logging requires an OmegaConf object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(NAME_MAP))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="setting", choices=["base", "setting"])
    parser.add_argument("--mode", default=None, choices=["tiger", "liger"],
                        help="defaults: tiger for method=base, liger for setting")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--set", action="append", default=[],
                        help="extra hydra overrides, e.g. TIGER.trainer.steps=100")
    args = parser.parse_args()
    if args.mode is None:
        args.mode = "tiger" if args.method == "base" else "liger"
    name = NAME_MAP[args.dataset]

    os.chdir(LIGER)  # LIGER's set_dir and writers use repo-relative paths
    from run import set_dir, set_seed  # noqa: E402  (LIGER repo modules)
    from ID_generation.train_rqvae import train as train_sid  # noqa: E402
    from ID_generation.utils import (  # noqa: E402
        process_data_split,
        process_embeddings,
    )
    from src.training import train_tiger  # noqa: E402

    config = build_config(args, name)
    device = (
        f"cuda:{args.device_id}"
        if args.device_id >= 0
        else "cpu"
    )
    set_seed(args.seed)

    path_config = set_dir(config)
    config = path_config.set_config(config)
    config["logging"]["project"] = "liger"

    processed = LIGER / "ID_generation" / "preprocessing" / "processed"
    data_file = str(processed / f"{name}.txt")
    id2meta_file = str(
        processed / f"{name}_{FEATURES}_amazon_id2meta.json")
    # LIGER's preprocessing is bypassed; the three files above come from
    # scripts.baselines.prepare_liger_data on the canonical splits.

    train_config = {
        **config["dataset"],
        **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]},
    }
    method_config = {
        **config["method"],
        **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]},
    }

    id_split, user_sequence = process_data_split(
        config, data_file, id2meta_file, is_steam=False)
    item_embedding = process_embeddings(
        config, device, id2meta_file, path_config.embedding_save_path).to(device)

    train_sid(config, device, item_embedding, id_split, path_config.id_save_location)
    train_tiger(
        config,
        train_config,
        method_config,
        id_split,
        user_sequence,
        item_embedding,
        path_config.id_save_location,
        device=device,
    )
    print(f"[liger] done: {name} seed{args.seed} method={args.method} "
          f"-> {config['output_path']}")


if __name__ == "__main__":
    main()
