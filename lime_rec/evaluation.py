"""Shared loading, scoring, and ranking utilities for evaluation commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .data import Dataset, load_dataset
from .experts import ItemCFExpert, LLMSemanticExpert
from .models import SASRec

MASKED_SCORE = -1e9


def load_configured_dataset(config_path: str | Path) -> Dataset:
    """Load a dataset from the repository's JSON configuration format."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    metadata = config.get("metadata_path")
    if metadata and not Path(metadata).exists():
        metadata = None
    return load_dataset(
        name=config["name"],
        interactions_path=config["interactions_path"],
        metadata_path=metadata,
        min_user_interactions=config.get("min_user_interactions", 5),
        min_item_interactions=config.get("min_item_interactions", 5),
        split_manifest=config.get("split_manifest"),
    )


def load_semantic_embeddings(
    path: str | Path, item_ids: Sequence[str]
) -> np.ndarray:
    """Align a saved embedding matrix to the dataset item ordering."""
    with np.load(path, allow_pickle=True) as archive:
        saved_ids = archive["item_ids"].tolist()
        embeddings = archive["embeddings"]
        by_id = dict(zip(saved_ids, embeddings))
        aligned = np.zeros((len(item_ids), embeddings.shape[1]), dtype=np.float32)
        for index, item_id in enumerate(item_ids):
            embedding = by_id.get(item_id)
            if embedding is not None:
                aligned[index] = embedding
    return aligned


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize unmasked catalog scores."""
    valid = scores[scores > -1e8]
    if valid.size == 0:
        return np.zeros_like(scores)
    return (scores - valid.min()) / (valid.max() - valid.min() + 1e-8)


def normalize_score_rows(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a batch of catalog-score rows."""
    valid = scores > -1e8
    row_min = np.where(valid, scores, np.inf).min(axis=1, keepdims=True)
    row_max = np.where(valid, scores, -np.inf).max(axis=1, keepdims=True)
    empty = ~valid.any(axis=1, keepdims=True)
    row_min = np.where(empty, 0.0, row_min)
    row_max = np.where(empty, 0.0, row_max)
    return (scores - row_min) / (row_max - row_min + 1e-8)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return top-k indices ordered by descending score."""
    if scores.size <= k:
        return np.argsort(-scores)
    candidates = np.argpartition(-scores, k)[:k]
    return candidates[np.argsort(-scores[candidates])]


def rank_of(scores: np.ndarray, target_index: int) -> int:
    """Return the zero-based rank of an item, using strict score ordering."""
    return int((scores > scores[target_index]).sum())


@dataclass
class SASRecScorer:
    """Checkpoint-backed SASRec scorer aligned to evaluation item indices."""

    model: SASRec
    maxlen: int
    item_to_model_index: Mapping[str, int]
    item_index: Mapping[str, int]
    num_items: int
    device: str = "cpu"
    evaluation_model_indices: np.ndarray | None = None
    evaluation_model_indices_tensor: torch.Tensor | None = None
    evaluation_valid_tensor: torch.Tensor | None = None

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, item_ids: Sequence[str], device: str = "cpu"
    ) -> "SASRecScorer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = SASRec(
            num_items=checkpoint["num_items"],
            hidden=checkpoint["hidden"],
            maxlen=checkpoint["maxlen"],
            num_layers=checkpoint["num_layers"],
            num_heads=checkpoint["num_heads"],
            dropout=0.0,
            architecture=checkpoint.get("architecture", "legacy"),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        item_to_model_index = checkpoint.get("item_to_idx") or {
            item_id: index + 1 for index, item_id in enumerate(item_ids)
        }
        item_index = {item_id: index for index, item_id in enumerate(item_ids)}
        evaluation_model_indices = np.asarray(
            [item_to_model_index.get(item_id, 0) for item_id in item_ids],
            dtype=np.int64,
        )
        evaluation_valid = evaluation_model_indices > 0
        return cls(
            model=model,
            maxlen=checkpoint["maxlen"],
            item_to_model_index=item_to_model_index,
            item_index=item_index,
            num_items=len(item_ids),
            device=device,
            evaluation_model_indices=evaluation_model_indices,
            evaluation_model_indices_tensor=torch.as_tensor(
                evaluation_model_indices, dtype=torch.long, device=device
            ),
            evaluation_valid_tensor=torch.as_tensor(
                evaluation_valid, dtype=torch.bool, device=device
            ),
        )

    @torch.inference_mode()
    def score(self, history: Sequence[str]) -> np.ndarray:
        """Score the complete catalog for one interaction history."""
        sequence = [self.item_to_model_index.get(item_id, 0) for item_id in history]
        sequence = [index for index in sequence if index][-self.maxlen:]
        if not sequence:
            return np.full(self.num_items, MASKED_SCORE, dtype=np.float32)
        padded = [0] * (self.maxlen - len(sequence)) + sequence
        logits = self.model.predict(torch.tensor([padded], dtype=torch.long, device=self.device))[0].detach().cpu().numpy()
        scores = np.full(self.num_items, MASKED_SCORE, dtype=np.float32)
        for item_id, model_index in self.item_to_model_index.items():
            evaluation_index = self.item_index.get(item_id)
            if evaluation_index is not None and 0 <= model_index < logits.shape[0]:
                scores[evaluation_index] = logits[model_index]
        return scores

    @torch.inference_mode()
    def score_batch(self, histories: Sequence[Sequence[str]]) -> np.ndarray:
        """Score the complete catalog for a batch of interaction histories."""
        padded_rows: list[list[int]] = []
        nonempty: list[bool] = []
        for history in histories:
            sequence = [self.item_to_model_index.get(item_id, 0) for item_id in history]
            sequence = [index for index in sequence if index][-self.maxlen :]
            nonempty.append(bool(sequence))
            padded_rows.append([0] * (self.maxlen - len(sequence)) + sequence)

        if not padded_rows:
            return np.empty((0, self.num_items), dtype=np.float32)
        sequence_tensor = torch.tensor(padded_rows, dtype=torch.long, device=self.device)
        logits = self.model.predict(sequence_tensor)
        scores = torch.full(
            (len(histories), self.num_items),
            MASKED_SCORE,
            dtype=logits.dtype,
            device=self.device,
        )
        valid = self.evaluation_valid_tensor
        model_indices = self.evaluation_model_indices_tensor
        scores[:, valid] = logits[:, model_indices[valid]]
        if not all(nonempty):
            empty = torch.tensor(nonempty, device=self.device).logical_not()
            scores[empty] = MASKED_SCORE
        return scores.cpu().numpy()


@dataclass
class ExpertEvaluator:
    """Reusable three-expert evaluation context for one configured dataset."""

    dataset: Dataset
    sasrec: SASRecScorer
    itemcf: ItemCFExpert
    semantic: LLMSemanticExpert
    mask_history: bool = True
    device: str = "cpu"
    semantic_catalog_tensor: torch.Tensor | None = None

    @classmethod
    def from_paths(
        cls,
        config_path: str | Path,
        sasrec_path: str | Path,
        itemcf_path: str | Path,
        semantic_path: str | Path,
        device: str = "cpu",
        mask_history: bool = True,
    ) -> "ExpertEvaluator":
        dataset = load_configured_dataset(config_path)
        embeddings = load_semantic_embeddings(semantic_path, dataset.item_ids)
        semantic = LLMSemanticExpert(
            item_ids=dataset.item_ids,
            item_text={},
            embeddings=embeddings,
            recency_decay=0.1,
        )
        return cls(
            dataset=dataset,
            sasrec=SASRecScorer.from_checkpoint(sasrec_path, dataset.item_ids, device),
            itemcf=ItemCFExpert(str(itemcf_path), dataset.item_ids),
            mask_history=mask_history,
            device=device,
            semantic=semantic,
            semantic_catalog_tensor=torch.as_tensor(
                semantic.item_embeddings_norm, dtype=torch.float32, device=device
            ),
        )

    def score(self, user_id: str, history: Sequence[str]) -> tuple[np.ndarray, ...]:
        """Return masked SASRec, ItemCF, and semantic catalog scores."""
        scores = (
            self.sasrec.score(history),
            self.itemcf.score(user_id, history),
            self.semantic.score(user_id, history),
        )
        if self.mask_history:
            for item_id in history:
                index = self.sasrec.item_index.get(item_id)
                if index is not None:
                    for expert_scores in scores:
                        expert_scores[index] = MASKED_SCORE
        return scores

    def normalized_scores(
        self, user_id: str, history: Sequence[str]
    ) -> tuple[np.ndarray, ...]:
        """Return masked and per-expert normalized scores."""
        return tuple(normalize_scores(scores) for scores in self.score(user_id, history))

    @torch.inference_mode()
    def score_batch(
        self, user_ids: Sequence[str], histories: Sequence[Sequence[str]]
    ) -> tuple[np.ndarray, ...]:
        """Return expert catalog scores for a batch of users."""
        sasrec_scores = self.sasrec.score_batch(histories)
        itemcf_scores = np.stack(
            [
                self.itemcf.score(user_id, list(history))
                for user_id, history in zip(user_ids, histories)
            ]
        )
        user_embeddings = np.stack(
            [self.semantic.user_embedding(list(history)) for history in histories]
        ).astype(np.float32, copy=False)
        norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True) + 1e-8
        normalized_users = torch.as_tensor(
            user_embeddings / norms, dtype=torch.float32, device=self.device
        )
        semantic_scores = (
            normalized_users @ self.semantic_catalog_tensor.T
        ).cpu().numpy()

        scores = (sasrec_scores, itemcf_scores, semantic_scores)
        if self.mask_history:
            for row, history in enumerate(histories):
                indices = [
                    self.sasrec.item_index[item_id]
                    for item_id in history
                    if item_id in self.sasrec.item_index
                ]
                if indices:
                    for expert_scores in scores:
                        expert_scores[row, indices] = MASKED_SCORE
        return scores
