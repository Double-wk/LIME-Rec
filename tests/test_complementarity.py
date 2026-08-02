import numpy as np

from scripts.evaluation.analyze_complementarity import hit_set_decomposition
from scripts.evaluation.dump_per_user_metrics import spearman_full


def test_spearman_full_uses_average_ranks_for_ties():
    a = np.array([0.0, 0.0, 1.0, 2.0])
    b = np.array([0.0, 0.0, 2.0, 1.0])

    assert np.isclose(spearman_full(a, b, np.ones(4, dtype=bool)), 7.0 / 9.0)


def test_spearman_full_honors_mask():
    a = np.array([99.0, 1.0, 2.0, 3.0])
    b = np.array([-99.0, 1.0, 2.0, 3.0])
    mask = np.array([False, True, True, True])

    assert np.isclose(spearman_full(a, b, mask), 1.0)


def test_spearman_full_returns_zero_for_constant_scores():
    a = np.zeros(4)
    b = np.arange(4.0)

    assert spearman_full(a, b, np.ones(4, dtype=bool)) == 0.0


def test_hit_set_decomposition_counts_all_eight_singleton_subsets():
    patterns = [
        (False, False, False), (True, False, False),
        (False, True, False), (False, False, True),
        (True, True, False), (True, False, True),
        (False, True, True), (True, True, True),
    ]
    entries = []
    for sas, icf, sem in patterns:
        entries.append({"experts": {
            "SASRec": {"target_rank": 0 if sas else 10},
            "ItemCF": {"target_rank": 0 if icf else 10},
            "Semantic": {"target_rank": 0 if sem else 10},
        }})

    actual = hit_set_decomposition(entries, k=10)

    assert actual == {
        "NONE": 12.5, "only_SAS": 12.5, "only_ICF": 12.5,
        "only_SEM": 12.5, "SAS_ICF": 12.5, "SAS_SEM": 12.5,
        "ICF_SEM": 12.5, "ALL": 12.5,
    }
