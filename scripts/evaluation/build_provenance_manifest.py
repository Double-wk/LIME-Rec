"""Create a hash-backed manifest for formal experiment inputs.

The manifest is deliberately generated only from ``output``.  It records
the training metadata embedded in every SASRec checkpoint plus hashes of every
formal model asset and training log, so results never need to rely on mutable
paths under ``outputs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_entry(path: Path, root: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    keys = (
        "architecture", "loss", "label_smoothing", "mask_mode", "mask_ratio",
        "mask_ce_weight", "consistency_weight", "seed", "validation_seed",
        "batch_size", "epochs", "patience", "eval_every", "eval_batch_size",
        "validation_mask_history", "torch_threads", "train_complete", "hidden",
        "maxlen", "num_layers", "num_heads", "dropout", "num_items",
    )
    entry = {key: checkpoint.get(key) for key in keys}
    entry.update({
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "type": "sasrec_checkpoint",
    })
    return entry


def static_asset_entry(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "type": "static_expert_asset",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output")
    parser.add_argument("--out", default="output/provenance/checkpoint_manifest.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    models = root / "models"
    logs = root / "logs" / "training"
    checkpoints, assets = [], []
    for path in sorted(models.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".pt":
            checkpoints.append(checkpoint_entry(path, root))
        else:
            assets.append(static_asset_entry(path, root))
    log_entries = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(logs.glob("*.log"))
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "worktree_dirty": bool(git_value(["git", "status", "--porcelain"])),
        "checkpoints": checkpoints,
        "static_assets": assets,
        "training_logs": log_entries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] checkpoints={len(checkpoints)} assets={len(assets)} -> {out}")


if __name__ == "__main__":
    main()
