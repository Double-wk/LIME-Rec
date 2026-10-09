"""Candidate-restricted wrappers around the existing expert evaluator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lime_rec.evaluation import ExpertEvaluator


def score_hash(scores: Sequence[float]) -> str:
    values = np.asarray(scores, dtype="<f4")
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass
class ExpertTool:
    name: str
    evaluator: ExpertEvaluator

    def query(self, user_id: str, history: Sequence[str], candidate_ids: Sequence[str]) -> dict:
        names = {"sasrec": 0, "itemcf": 1, "semantic": 2}
        if self.name not in names:
            raise ValueError(f"unknown expert tool: {self.name}")
        catalog_scores = self.evaluator.normalized_scores(user_id, history)[names[self.name]]
        indices = [self.evaluator.sasrec.item_index[item] for item in candidate_ids]
        values = np.asarray(catalog_scores[indices], dtype=np.float32)
        order = sorted(range(len(candidate_ids)), key=lambda i: (-float(values[i]), str(candidate_ids[i])))
        ranks = {index: rank for rank, index in enumerate(order, start=1)}
        return {
            "tool": self.name,
            "score_hash": score_hash(values),
            "items": [
                {"item_id": str(item), "score": float(values[i]),
                 "rank_within_candidate_pool": ranks[i]}
                for i, item in enumerate(candidate_ids)
            ],
        }


class SASRecTool(ExpertTool):
    def __init__(self, evaluator: ExpertEvaluator):
        super().__init__("sasrec", evaluator)


class ItemCFTool(ExpertTool):
    def __init__(self, evaluator: ExpertEvaluator):
        super().__init__("itemcf", evaluator)


class SemanticTool(ExpertTool):
    def __init__(self, evaluator: ExpertEvaluator):
        super().__init__("semantic", evaluator)
