"""Validation-fitted recovery gate restricted to the shared candidate pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from scripts.evaluation.run_repeat_aware_gate import RepeatAwareGate, features


def candidate_features(scores: torch.Tensor, history_mask: torch.Tensor,
                       history_lengths: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 3 or scores.shape[1] != 3:
        raise ValueError("scores must have shape [B, 3, C]")
    if scores.shape[2] < 5:
        raise ValueError("candidate pool must contain at least five items")
    return features(scores, history_lengths, history_mask)


@dataclass
class CandidateRecovery:
    max_penalty: float = 0.10
    score_scale: float = 20.0
    initial_weights: tuple[float, float, float] = (0.60, 0.15, 0.25)
    device: str = "cpu"

    def __post_init__(self):
        self.gate = RepeatAwareGate(14, self.initial_weights, self.max_penalty).to(self.device)
        self.fitted_split: str | None = None

    def fit(self, scores: np.ndarray, history_mask: np.ndarray, history_lengths: np.ndarray,
            target_indices: np.ndarray, *, split: str, epochs: int = 4,
            lr: float = 0.02, l2: float = 0.001) -> list[float]:
        if split != "validation":
            raise RuntimeError("recovery fitting is validation-only")
        tensors = (
            torch.as_tensor(scores, dtype=torch.float32, device=self.device),
            torch.as_tensor(history_mask, dtype=torch.bool, device=self.device),
            torch.as_tensor(history_lengths, dtype=torch.float32, device=self.device),
            torch.as_tensor(target_indices, dtype=torch.long, device=self.device),
        )
        optimizer = torch.optim.AdamW(self.gate.parameters(), lr=lr, weight_decay=l2)
        losses: list[float] = []
        self.gate.train()
        for _ in range(epochs):
            expert_scores, mask, lengths, targets = tensors
            weights, penalty = self.gate(candidate_features(expert_scores, mask, lengths))
            fused = (weights.unsqueeze(2) * expert_scores).sum(1) - penalty.unsqueeze(1) * mask
            loss = torch.nn.functional.cross_entropy(self.score_scale * fused, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.fitted_split = split
        return losses

    def fit_records(self, records: Sequence[dict], *, split: str, epochs: int = 4,
                    lr: float = 0.02, l2: float = 0.001) -> list[float]:
        """Fit variable-sized candidate pools without padding their statistics."""
        if split != "validation":
            raise RuntimeError("recovery fitting is validation-only")
        usable = [record for record in records if record.get("target_index") is not None]
        if not usable:
            raise ValueError("no validation targets occur naturally in their candidate pools")
        optimizer = torch.optim.AdamW(self.gate.parameters(), lr=lr, weight_decay=l2)
        losses: list[float] = []
        self.gate.train()
        for _ in range(epochs):
            for record in usable:
                scores = torch.as_tensor(record["expert_scores"], dtype=torch.float32,
                                         device=self.device).unsqueeze(0)
                mask = torch.as_tensor(record["history_mask"], dtype=torch.bool,
                                       device=self.device).unsqueeze(0)
                lengths = torch.tensor([record["history_length"]], dtype=torch.float32,
                                       device=self.device)
                target = torch.tensor([record["target_index"]], dtype=torch.long,
                                      device=self.device)
                weights, penalty = self.gate(candidate_features(scores, mask, lengths))
                fused = (weights.unsqueeze(2) * scores).sum(1) - penalty.unsqueeze(1) * mask
                loss = torch.nn.functional.cross_entropy(self.score_scale * fused, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        self.fitted_split = split
        return losses

    @torch.inference_mode()
    def score(self, scores: np.ndarray, history_mask: np.ndarray,
              history_lengths: np.ndarray) -> np.ndarray:
        if self.fitted_split != "validation":
            raise RuntimeError("recovery gate must be fitted on validation before scoring")
        expert_scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(history_mask, dtype=torch.bool, device=self.device)
        lengths = torch.as_tensor(history_lengths, dtype=torch.float32, device=self.device)
        self.gate.eval()
        weights, penalty = self.gate(candidate_features(expert_scores, mask, lengths))
        return ((weights.unsqueeze(2) * expert_scores).sum(1) - penalty.unsqueeze(1) * mask).cpu().numpy()

    def state_dict(self) -> dict:
        return {key: value.detach().cpu() for key, value in self.gate.state_dict().items()}
