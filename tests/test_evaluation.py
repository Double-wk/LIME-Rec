import numpy as np

from lime_rec.evaluation import normalize_scores, rank_of, top_k_indices


def test_normalize_scores_ignores_masked_values():
    scores = np.array([-1e9, 2.0, 4.0], dtype=np.float32)

    normalized = normalize_scores(scores)

    np.testing.assert_allclose(normalized[1:], [0.0, 1.0])
    assert normalized[0] < -1e8


def test_ranking_helpers_use_descending_scores():
    scores = np.array([0.2, 0.9, 0.5, 0.1], dtype=np.float32)

    np.testing.assert_array_equal(top_k_indices(scores, 2), [1, 2])
    assert rank_of(scores, 2) == 1
