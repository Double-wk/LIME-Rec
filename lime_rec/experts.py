"""The two non-sequential experts used by LIME-Rec.

SASRec checkpoint loading lives in :mod:`lime_rec.evaluation`, so this module
contains only the ItemCF and semantic scoring components used by the public
pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class ItemCFExpert:
    """Sparse ItemCF scorer backed by a model from ``build_itemcf``."""

    name = "itemcf"

    def __init__(self, model_path: str, item_ids: List[str]):
        path = Path(model_path)
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load ItemCF model at {path}: {exc}") from exc
        if not isinstance(model, dict) or not isinstance(model.get("neighbors"), dict):
            raise ValueError(f"ItemCF model at {path} has no valid 'neighbors' mapping")
        if not model["neighbors"]:
            raise ValueError(f"ItemCF model at {path} contains no neighbours")

        self.item_ids = item_ids
        self.item_index = {item_id: index for index, item_id in enumerate(item_ids)}
        self.num_items = len(item_ids)
        self.history_window = int(model.get("scoring_history_length", 20))
        if self.history_window < 1:
            raise ValueError("ItemCF scoring_history_length must be positive")

        self.neighbors: Dict[str, np.ndarray] = {}
        self.neighbor_scores: Dict[str, np.ndarray] = {}
        for source, related in model["neighbors"].items():
            if not isinstance(related, list):
                continue
            indices: list[int] = []
            scores: list[float] = []
            for row in related:
                if not isinstance(row, dict):
                    continue
                target_index = self.item_index.get(str(row.get("item_id", "")))
                if target_index is None:
                    continue
                try:
                    score = float(row["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                indices.append(target_index)
                scores.append(score)
            if indices:
                self.neighbors[str(source)] = np.asarray(indices, dtype=np.int64)
                self.neighbor_scores[str(source)] = np.asarray(scores, dtype=np.float32)

        if not self.neighbors:
            raise ValueError(
                f"ItemCF model at {path} has no neighbours aligned to the current catalog"
            )

    def score(self, user_id: str, history: List[str]) -> np.ndarray:
        """Score items from the most recent history window with linear recency."""
        scores = np.zeros(self.num_items, dtype=np.float32)
        window = history[-self.history_window :]
        for offset, item_id in enumerate(window, start=1):
            indices = self.neighbors.get(item_id)
            if indices is None:
                continue
            recency_weight = offset / len(window)
            scores[indices] += recency_weight * self.neighbor_scores[item_id]
        return scores


class LLMSemanticExpert:
    """Cosine similarity between a recency-weighted user and item embedding."""

    name = "semantic"

    def __init__(
        self,
        item_ids: List[str],
        item_text: Dict[str, str],
        embedding_dim: int = 128,
        recency_decay: float = 0.1,
        embeddings: Optional[np.ndarray] = None,
    ):
        self.item_ids = item_ids
        self.item_index = {item_id: index for index, item_id in enumerate(item_ids)}
        self.num_items = len(item_ids)
        self.recency_decay = recency_decay

        if embeddings is not None:
            if embeddings.shape[0] != self.num_items:
                raise ValueError("Semantic embedding rows must align with item_ids")
            self.item_embeddings = embeddings.astype(np.float32)
        else:
            self.item_embeddings = _hash_text_embedding(
                [item_text.get(item_id, item_id) for item_id in item_ids], embedding_dim
            )
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True) + 1e-8
        self.item_embeddings_norm = self.item_embeddings / norms

    def user_embedding(self, history: List[str]) -> np.ndarray:
        if not history:
            return np.zeros(self.item_embeddings.shape[1], dtype=np.float32)
        weights: list[float] = []
        vectors: list[np.ndarray] = []
        for position, item_id in enumerate(history):
            item_index = self.item_index.get(item_id)
            if item_index is None:
                continue
            weights.append(math.exp(-self.recency_decay * (len(history) - 1 - position)))
            vectors.append(self.item_embeddings[item_index])
        if not vectors:
            return np.zeros(self.item_embeddings.shape[1], dtype=np.float32)
        weight_matrix = np.asarray(weights, dtype=np.float32)[:, None]
        return (np.stack(vectors, axis=0) * weight_matrix).sum(axis=0) / max(
            weight_matrix.sum(), 1e-8
        )

    def score(self, user_id: str, history: List[str]) -> np.ndarray:
        user_embedding = self.user_embedding(history)
        normalized_user = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)
        return self.item_embeddings_norm @ normalized_user


def _hash_text_embedding(texts: List[str], dim: int) -> np.ndarray:
    """Create a deterministic fallback representation when no encoder is supplied."""
    pattern = re.compile(r"[A-Za-z0-9]+")
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = pattern.findall(text.lower()) or [str(row)]
        for token, count in Counter(tokens).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % dim
            sign = 1.0 if value & 1 else -1.0
            matrix[row, bucket] += sign * (1.0 + math.log1p(count))
    return matrix
