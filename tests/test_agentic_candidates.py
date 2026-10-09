import inspect

import numpy as np

from lime_rec.agentic.candidates import build_candidate_pool


def test_candidate_builder_never_uses_target():
    assert "target" not in inspect.signature(build_candidate_pool).parameters


def test_candidate_pool_deterministic_and_shared_across_conditions():
    ids = ["b", "a", "c"]
    scores = {"sasrec": np.array([1.0, 1.0, 0.0]),
              "itemcf": np.array([0.0, 0.0, 2.0]),
              "semantic": np.array([0.0, 0.0, 0.0])}
    first = build_candidate_pool("u", ids, scores, 2)
    second = build_candidate_pool("u", ids, scores, 2)
    assert first == second
    assert first.item_ids == ("a", "b", "c")
