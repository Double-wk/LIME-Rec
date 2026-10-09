"""Parsing and auditing helpers for the official GRAM release."""

from __future__ import annotations

import hashlib
from pathlib import Path


def read_user_sequences(path: str | Path) -> dict[str, list[str]]:
    sequences: dict[str, list[str]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                sequences[parts[0]] = parts[1:]
    return sequences


def read_semantic_mapping(path: str | Path) -> tuple[dict[str, str], int]:
    """Return semantic-id -> raw-id mapping and collision count."""
    reverse: dict[str, str] = {}
    collisions = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t", 1)
            if len(parts) != 2:
                parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"invalid GRAM mapping row: {stripped[:80]}")
            raw_id, semantic_id = parts[0], parts[1].strip()
            if semantic_id in reverse and reverse[semantic_id] != raw_id:
                collisions += 1
            else:
                reverse[semantic_id] = raw_id
    return reverse, collisions


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
