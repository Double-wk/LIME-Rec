import math

import pytest

from lime_rec.controlled_metrics import evaluate_rankings, stable_unique_ranking


def test_controlled_metrics_known_example():
    metrics = evaluate_rankings(
        {"u1": ["x", "target"], "u2": ["x", "y"]},
        {"u1": "target", "u2": "target"},
        ks=(1, 2),
    )
    assert metrics["R@1"] == 0.0
    assert metrics["R@2"] == 0.5
    assert metrics["N@2"] == pytest.approx(0.5 / math.log2(3))


def test_controlled_metrics_never_drops_missing_users():
    with pytest.raises(ValueError, match="users differ"):
        evaluate_rankings({}, {"u": "target"})


def test_stable_unique_ranking_counts_duplicates():
    assert stable_unique_ranking(["a", "b", "a"]) == (["a", "b"], 1)
