"""Build pre-computed semantic embeddings for each item in a dataset.

Uses sentence-transformers (all-MiniLM-L6-v2 by default), MPS-accelerated on Apple Silicon.
One-time offline cost. Outputs are reused by SemanticCFExpert at inference (matrix lookup).

Item text = title | brand | category | description (concatenated with separators).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from lime_rec.evaluation import load_configured_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="HuggingFace model id for the encoder")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=None,
                   help="Cap model max_seq_length (defaults to model default; lower = less memory)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                   help="Device for inference. 'auto' prefers MPS, then CUDA, then CPU.")
    p.add_argument("--out", default=None,
                   help="Output .npz path; defaults to outputs/embeddings/<name>_minilm.npz")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    name = cfg["name"]
    out_path = args.out or f"outputs/embeddings/{name}_minilm.npz"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    ds = load_configured_dataset(args.config)
    print(f"[data] {name}: items={ds.num_items}", flush=True)

    texts = [ds.item_text.get(item_id, item_id) for item_id in ds.item_ids]
    sample = texts[:3]
    for s in sample:
        print(f"  sample text: {s[:120]}", flush=True)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[device] {device}", flush=True)
    model = SentenceTransformer(args.model, device=device)
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length
    print(f"[model] {args.model} dim={model.get_sentence_embedding_dimension()} "
          f"max_seq_length={model.max_seq_length}", flush=True)

    emb = model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    print(f"[emb] shape={emb.shape} dtype={emb.dtype}", flush=True)

    np.savez_compressed(
        out_path,
        item_ids=np.array(ds.item_ids, dtype=object),
        embeddings=emb.astype(np.float32),
    )
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
