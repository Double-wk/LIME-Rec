"""SASRec training, defaulting to the original ICDM'18 paper recipe.

The default objective samples one unseen negative per non-padding position and
uses binary cross entropy. Full-catalog CE remains available only as an
explicit SASRec+ control via --loss ce.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lime_rec.models import SASRec

class ExpertDisagreementMasker:
    """Training-only ItemCF/semantic disagreement perturbation."""

    def __init__(self, itemcf_path, semantic_emb_path, item_ids):
        raw = json.loads(Path(itemcf_path).read_text(encoding="utf-8"))
        index = {item_id: pos + 1 for pos, item_id in enumerate(item_ids)}
        self.neighbors = {}
        self.cache = {}
        for source, rows in raw.get("neighbors", {}).items():
            src = index.get(str(source))
            if src is None:
                continue
            ids, vals = [], []
            for row in rows:
                idx = index.get(str(row.get("item_id")))
                if idx is not None:
                    ids.append(idx)
                    vals.append(float(row["score"]))
            if ids:
                self.neighbors[src] = (np.asarray(ids), np.asarray(vals))
        with np.load(semantic_emb_path, allow_pickle=False) as archive:
            emb = archive["embeddings"].astype(np.float32)
        if emb.shape[0] != len(item_ids):
            raise ValueError("Semantic embeddings do not align with item ids")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        self.embeddings = torch.from_numpy(np.vstack([np.zeros((1, emb.shape[1]), dtype=np.float32), emb]))

    def mask(self, input_ids, ratio):
        out = input_ids.clone()
        for row in range(input_ids.shape[0]):
            valid = torch.nonzero(input_ids[row] != 0, as_tuple=False).flatten().tolist()
            if len(valid) < 2:
                continue
            key = tuple(int(input_ids[row, position]) for position in valid)
            cached = self.cache.get(key)
            if cached is not None:
                out[row, torch.tensor(cached, device=input_ids.device)] = 0
                continue
            candidates, prefix, values = valid[1:], [int(input_ids[row, valid[0]])], []
            for position in candidates:
                target = int(input_ids[row, position])
                semantic = float((self.embeddings[prefix].mean(0) * self.embeddings[target]).sum())
                cf = 0.0
                for source in prefix:
                    related = self.neighbors.get(source)
                    if related is None:
                        continue
                    ids, vals = related
                    match = np.where(ids == target)[0]
                    if match.size:
                        cf += float(vals[match].sum())
                values.append(semantic - cf)
                prefix.append(target)
            score = torch.tensor(values, device=input_ids.device)
            score = (score - score.mean()) / (score.std(unbiased=False) + 1e-6)
            count = max(1, math.ceil(len(candidates) * ratio))
            choose = score.abs().topk(count).indices.tolist()
            chosen_positions = [candidates[i] for i in choose]
            self.cache[key] = chosen_positions
            out[row, torch.tensor(chosen_positions, device=input_ids.device)] = 0
        return out

def random_mask(input_ids, ratio):
    valid = input_ids != 0
    first = valid.float().argmax(dim=1)
    valid[torch.arange(input_ids.shape[0], device=input_ids.device), first] = False
    return input_ids.masked_fill(valid & (torch.rand_like(input_ids.float()) < ratio), 0)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SeqDataset(Dataset):
    """Multi-position training: for each user, create input/target pairs at
    every position in the sequence."""

    def __init__(self, sequences, maxlen: int, num_items: int):
        self.maxlen = maxlen
        self.num_items = num_items
        self.data = []
        self.seen = []
        for seq in sequences:
            if len(seq) < 2:
                continue
            self.data.append(seq)
            self.seen.append(set(seq))
        self.masked_inputs = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Truncate to maxlen+1 (last item is the final target).
        seq = seq[-(self.maxlen + 1):]
        input_ids = seq[:-1]
        target_ids = seq[1:]
        # Pad from the left.
        pad_len = self.maxlen - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_ids = [0] * pad_len + target_ids
        inputs = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(target_ids, dtype=torch.long)
        negatives = torch.zeros_like(targets)
        seen = self.seen[idx]
        if len(seen) >= self.num_items:
            raise ValueError("Cannot sample an unseen item: user has seen the full catalog")
        for position, target in enumerate(target_ids):
            if target == 0:
                continue
            negative = random.randint(1, self.num_items)
            while negative in seen:
                negative = random.randint(1, self.num_items)
            negatives[position] = negative
        if self.masked_inputs is not None:
            return inputs, targets, negatives, self.masked_inputs[idx]
        return inputs, targets, negatives


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _resolve_device(device_arg: str, torch_threads: int | None = None) -> torch.device:
    """Resolve user-facing device string to a torch.device.

    ``auto`` prefers MPS (Apple Silicon GPU) when available, falls back to CPU.
    Set PyTorch CPU thread pools. ``torch_threads`` lets independent seed runs
    share a CPU host without each claiming every visible core.
    """
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            chosen = "mps"
        elif torch.cuda.is_available():
            chosen = "cuda"
        else:
            chosen = "cpu"
    else:
        chosen = device_arg
    if chosen == "cpu":
        n = torch_threads or os.cpu_count() or 1
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 2))
    return torch.device(chosen)


def _seed_everything(seed: int) -> None:
    """Seed the libraries used by this training command."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sasrec(
    train_seqs,
    num_items: int,
    hidden: int = 50,
    maxlen: int = 50,
    num_layers: int = 2,
    num_heads: int = 1,
    dropout: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    epochs: int = 200,
    patience: int = 20,
    valid_seqs=None,
    valid_targets=None,
    all_item_ids=None,
    ckpt_path: str | None = None,
    ckpt_meta: dict | None = None,
    device: torch.device | None = None,
    eval_every: int = 10,
    eval_batch_size: int = 128,
    loss_name: str = "paper",
    mask_mode: str = "none",
    mask_ratio: float = 0.2,
    mask_ce_weight: float = 0.3,
    consistency_weight: float = 0.1,
    masker: ExpertDisagreementMasker | None = None,
    validation_mask_history: bool = True,
):
    if device is None:
        device = torch.device("cpu")
    print(f"[SASRec] device={device}", flush=True)
    # Historical checkpoint labels: "paper" selects the original ICDM'18-style
    # block, while "legacy" selects the pre-normalized block used by the main CE
    # experiments. Keep these labels stable for checkpoint provenance.
    architecture = "paper" if loss_name == "paper" else "legacy"
    model = SASRec(num_items, hidden, maxlen, num_layers, num_heads, dropout,
                   architecture=architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = SeqDataset(train_seqs, maxlen, num_items)
    if mask_mode == "disagreement":
        if masker is None:
            raise ValueError("disagreement masking requires ItemCF and semantic inputs")
        print("[SASRec] precomputing disagreement masks once", flush=True)
        dataset.masked_inputs = [
            masker.mask(dataset[idx][0].unsqueeze(0), mask_ratio).squeeze(0)
            for idx in range(len(dataset))
        ]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    best_metric = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in loader:
            if mask_mode == "disagreement":
                input_ids, target_ids, negative_ids, masked_input = batch
            else:
                input_ids, target_ids, negative_ids = batch
                masked_input = None
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            negative_ids = negative_ids.to(device)
            hidden_states = model(input_ids)
            mask = target_ids != 0
            targets_flat = target_ids[mask]
            if loss_name == "paper":
                positive_logits = (hidden_states[mask] * model.item_emb(target_ids)[mask]).sum(-1)
                negative_logits = (hidden_states[mask] * model.item_emb(negative_ids)[mask]).sum(-1)
                loss = (
                    F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
                    + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
                )
            elif loss_name == "ce":
                # The loss is defined only at non-padding targets.  Projecting
                # the whole ``batch_size x maxlen`` tensor first is exactly
                # equivalent, but wastes a full-vocabulary matmul on padding
                # positions (most sequences are shorter than ``maxlen``).
                logits_flat = hidden_states[mask] @ model.item_emb.weight.T
                loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=0.1)
            elif loss_name == "bce":
                logits_flat = hidden_states[mask] @ model.item_emb.weight.T
                target_matrix = F.one_hot(
                    targets_flat, num_classes=num_items + 1
                ).to(dtype=logits_flat.dtype)
                # Padding index 0 is not a recommendable item and is excluded
                # from the full-catalog binary objective.
                loss = F.binary_cross_entropy_with_logits(
                    logits_flat[:, 1:], target_matrix[:, 1:]
                )
            else:
                raise ValueError(f"Unsupported loss: {loss_name}")
            if mask_mode != "none":
                if mask_mode == "random":
                    masked_input = random_mask(input_ids, mask_ratio)
                elif mask_mode == "disagreement" and masked_input is not None:
                    masked_input = masked_input.to(device)
                else:
                    raise ValueError("disagreement masking requires ItemCF and semantic inputs")
                masked_hidden = model(masked_input)
                masked_logits = masked_hidden[mask] @ model.item_emb.weight.T
                masked_ce = F.cross_entropy(masked_logits, targets_flat, label_smoothing=0.1)
                feature_loss = F.mse_loss(
                    masked_hidden[mask], hidden_states[mask].detach()
                )
                loss = loss + mask_ce_weight * masked_ce + consistency_weight * feature_loss
            optimizer.zero_grad()
            loss.backward()
            if loss_name != "paper":
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * targets_flat.shape[0]
            total_items += targets_flat.shape[0]

        avg_loss = total_loss / max(total_items, 1)

        # Validation on configurable interval.
        if valid_seqs is not None and epoch % eval_every == 0:
            metric = evaluate_recall(model, valid_seqs, valid_targets, maxlen, device, k=10,
                                     batch_size=eval_batch_size, mask_history=validation_mask_history)
            print(f"[SASRec] epoch={epoch} loss={avg_loss:.4f} val_R@10={metric:.4f}", flush=True)
            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                if ckpt_path is not None:
                    payload = {"state_dict": best_state, "epoch": epoch,
                               "val_metric": best_metric, "hidden": hidden,
                               "maxlen": maxlen, "num_layers": num_layers,
                               "num_heads": num_heads, "dropout": dropout,
                               "architecture": architecture, "loss": loss_name,
                               "num_items": num_items}
                    if ckpt_meta:
                        payload.update(ckpt_meta)
                    tmp_path = ckpt_path + ".tmp"
                    torch.save(payload, tmp_path)
                    Path(tmp_path).replace(ckpt_path)
            else:
                no_improve += eval_every
            if no_improve >= patience:
                print(f"[SASRec] early stop at epoch {epoch}", flush=True)
                break
        else:
            print(f"[SASRec] epoch={epoch} loss={avg_loss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_recall(model, seqs, targets, maxlen, device, k=10, batch_size=128, mask_history=True):
    """Batched validation: stack all (seq, target) into batches and score in one forward."""
    model.eval()
    pairs = [(s, t) for s, t in zip(seqs, targets) if s and t != 0]
    if not pairs:
        return 0.0
    hits = 0
    total = 0
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        B = len(chunk)
        inps = np.zeros((B, maxlen), dtype=np.int64)
        targets_t = np.zeros((B,), dtype=np.int64)
        for r, (seq, target) in enumerate(chunk):
            ids = seq[-maxlen:]
            pad = maxlen - len(ids)
            inps[r, pad:] = ids
            targets_t[r] = target
        inp = torch.from_numpy(inps).to(device)
        logits = model.predict(inp)  # [B, V]
        # Mask history items per row + pad index 0.
        for r, (seq, _) in enumerate(chunk):
            logits[r, 0] = -1e9
            if mask_history:
                for item in seq:
                    logits[r, item] = -1e9
        topk = logits.topk(k, dim=1).indices.cpu().numpy()  # [B, k]
        for r in range(B):
            if targets_t[r] in topk[r]:
                hits += 1
            total += 1
    return hits / max(total, 1)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", default="outputs/models/sasrec.pt")
    p.add_argument("--hidden", type=int, default=50)
    p.add_argument("--maxlen", type=int, default=50)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--loss", choices=["paper", "ce", "bce"], default="paper",
                   help="paper = original ICDM'18 block with one-negative BCE; ce/bce use the pre-normalized block.")
    p.add_argument("--mask-mode", choices=["none", "random", "disagreement"], default="none",
                   help="Training-only history perturbation; inference is always unmasked.")
    p.add_argument("--mask-ratio", type=float, default=0.2)
    p.add_argument("--mask-ce-weight", type=float, default=0.3)
    p.add_argument("--consistency-weight", type=float, default=0.1)
    p.add_argument("--itemcf-model", default=None,
                   help="Required only for --mask-mode disagreement.")
    p.add_argument("--semantic-emb", default=None,
                   help="Required only for --mask-mode disagreement.")
    p.add_argument("--validation-no-mask", action="store_true",
                   help="Use MHL-aligned no-history-filtering validation for checkpoint selection.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for initialization and data order.")
    p.add_argument("--validation-seed", type=int, default=0,
                   help="Fixed seed for the validation-user subsample used for early stopping.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                   help="auto = MPS if available else CPU.")
    p.add_argument("--eval-every", type=int, default=10,
                   help="Run validation every N epochs (default 10).")
    p.add_argument("--eval-batch-size", type=int, default=128,
                   help="Batch size for batched validation forward passes.")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="PyTorch CPU threads; defaults to all visible CPUs.")
    args = p.parse_args()
    if args.torch_threads is not None and args.torch_threads < 1:
        p.error("--torch-threads must be at least 1")
    _seed_everything(args.seed)
    torch_threads = args.torch_threads or os.cpu_count() or 1

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    from lime_rec.data import load_dataset
    ds = load_dataset(
        name=config["name"],
        interactions_path=config["interactions_path"],
        min_user_interactions=config.get("min_user_interactions", 5),
        min_item_interactions=config.get("min_item_interactions", 5),
        split_manifest=config.get("split_manifest"),
    )
    print(f"[data] users={ds.num_users} items={ds.num_items}", flush=True)

    # Build item index (1-indexed, 0=pad).
    item_to_idx = {iid: i + 1 for i, iid in enumerate(ds.item_ids)}
    num_items = len(ds.item_ids)

    # Build training sequences (all history + valid item for training).
    train_seqs = []
    valid_seqs = []
    valid_targets = []
    for u in ds.user_ids:
        history = ds.history_by_user.get(u, [])
        seq = [item_to_idx[i] for i in history if i in item_to_idx]
        valid_item = ds.valid_by_user.get(u)
        if config.get("split_manifest"):
            train_seqs.append(seq)
        if valid_item and valid_item in item_to_idx:
            # Training sequence includes everything up to (not including) valid.
            if not config.get("split_manifest"):
                train_seqs.append(seq)
            # Validation: input=seq, target=valid_item_idx.
            valid_seqs.append(seq)
            valid_targets.append(item_to_idx[valid_item])

    print(f"[SASRec] train_seqs={len(train_seqs)} valid_users={len(valid_seqs)}", flush=True)
    print(f"[SASRec] hidden={args.hidden} layers={args.layers} heads={args.heads} "
          f"maxlen={args.maxlen} dropout={args.dropout} lr={args.lr} "
          f"loss={args.loss} seed={args.seed} epochs={args.epochs} "
          f"torch_threads={torch_threads} validation_seed={args.validation_seed}", flush=True)
    if not 0.0 <= args.mask_ratio <= 1.0:
        p.error("--mask-ratio must be in [0, 1]")
    if args.mask_mode == "disagreement" and (not args.itemcf_model or not args.semantic_emb):
        p.error("--mask-mode disagreement requires --itemcf-model and --semantic-emb")
    if args.loss == "paper" and args.mask_mode != "none":
        p.error("The original paper recipe requires --mask-mode none")
    masker = (ExpertDisagreementMasker(args.itemcf_model, args.semantic_emb, ds.item_ids)
              if args.mask_mode == "disagreement" else None)

    # Subsample validation for speed. Keep this sample independent from the
    # training seed so multi-seed comparisons change model stochasticity, not
    # the held-out users used to select the checkpoint.
    val_subset = 2000
    if len(valid_seqs) > val_subset:
        rng = np.random.default_rng(args.validation_seed)
        idx = rng.choice(len(valid_seqs), val_subset, replace=False)
        valid_seqs_sub = [valid_seqs[i] for i in idx]
        valid_targets_sub = [valid_targets[i] for i in idx]
    else:
        valid_seqs_sub = valid_seqs
        valid_targets_sub = valid_targets

    # The best-validation checkpoint is an implementation detail.  The public
    # output path is written only after training returns, so an interrupted run
    # cannot be mistaken for a completed seed by a workflow that checks for it.
    best_path = args.output + ".best"
    model = train_sasrec(
        train_seqs=train_seqs,
        num_items=num_items,
        hidden=args.hidden,
        maxlen=args.maxlen,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        valid_seqs=valid_seqs_sub,
        valid_targets=valid_targets_sub,
        ckpt_path=best_path,
        ckpt_meta={"item_ids": ds.item_ids, "item_to_idx": item_to_idx},
        device=_resolve_device(args.device, args.torch_threads),
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        loss_name=args.loss,
        mask_mode=args.mask_mode,
        mask_ratio=args.mask_ratio,
        mask_ce_weight=args.mask_ce_weight,
        consistency_weight=args.consistency_weight,
        masker=masker,
        validation_mask_history=not args.validation_no_mask,
    )

    # Save model + metadata.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "item_ids": ds.item_ids,
        "item_to_idx": item_to_idx,
        "num_items": num_items,
        "hidden": args.hidden,
        "maxlen": args.maxlen,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "dropout": args.dropout,
        "architecture": "paper" if args.loss == "paper" else "legacy",
        "loss": args.loss,
        "label_smoothing": 0.1 if args.loss == "ce" else None,
        "mask_mode": args.mask_mode,
        "mask_ratio": args.mask_ratio,
        "mask_ce_weight": args.mask_ce_weight,
        "consistency_weight": args.consistency_weight,
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "eval_every": args.eval_every,
        "eval_batch_size": args.eval_batch_size,
        "validation_mask_history": not args.validation_no_mask,
        "torch_threads": torch_threads,
        "train_complete": True,
    }
    tmp_path = Path(str(output_path) + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)
    Path(best_path).unlink(missing_ok=True)
    print(f"[SASRec] saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
