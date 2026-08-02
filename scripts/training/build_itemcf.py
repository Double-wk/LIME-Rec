"""Build a training-only ItemCF model for one dataset configuration.

Example:
    python -m scripts.training.build_itemcf \
      --config configs/amazon_beauty.json \
      --out outputs/models/amazon_beauty_itemcf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lime_rec.evaluation import load_configured_dataset
from lime_rec.itemcf import build_itemcf_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Dataset JSON configuration.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: outputs/models/<dataset>_itemcf.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Maximum neighbours stored per item (default: 100).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Co-occurrence and scoring history window (default: 20).",
    )
    args = parser.parse_args()

    dataset = load_configured_dataset(args.config)
    model = build_itemcf_model(
        dataset_name=dataset.name,
        histories=dataset.history_by_user,
        item_ids=dataset.item_ids,
        top_k=args.top_k,
        window_size=args.window_size,
    )
    if not model["neighbors"]:
        raise ValueError(
            "No ItemCF neighbours were built. Check that the training histories "
            "contain at least two distinct items within the configured window."
        )

    output_path = Path(args.out or f"outputs/models/{dataset.name}_itemcf.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            model,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "[ItemCF] "
        f"dataset={dataset.name} users={model['num_training_users']} "
        f"interactions={model['num_training_interactions']} "
        f"items_with_neighbours={model['num_items_with_neighbours']} -> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
