import numpy as np
import pytest

from lime_rec.agentic.recovery import CandidateRecovery


def test_recovery_fit_never_reads_test_targets():
    recovery = CandidateRecovery()
    scores = np.zeros((1, 3, 5), dtype=np.float32)
    history = np.zeros((1, 5), dtype=bool)
    lengths = np.ones(1, dtype=np.float32)
    targets = np.zeros(1, dtype=np.int64)
    with pytest.raises(RuntimeError, match="validation-only"):
        recovery.fit(scores, history, lengths, targets, split="test")


def test_recovery_can_fit_validation_and_score():
    recovery = CandidateRecovery()
    scores = np.zeros((1, 3, 5), dtype=np.float32)
    scores[0, :, 1] = 1.0
    history = np.zeros((1, 5), dtype=bool)
    recovery.fit(scores, history, np.array([2.0]), np.array([1]), split="validation", epochs=1)
    assert recovery.score(scores, history, np.array([2.0])).shape == (1, 5)
