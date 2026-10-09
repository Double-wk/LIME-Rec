import pytest

from lime_rec.agentic.metrics import evaluate_agentic


def _record(user_id, target, candidates, ranking, format_failure=False):
    return {"user_id": user_id, "target_item_id": target, "candidate_ids": candidates,
            "ranking": ranking, "format_failure": format_failure}


def test_coverage_factorization_identity_holds():
    # u1: target in pool and hit; u2: target in pool and missed;
    # u3: target outside pool (can never be hit); u4: format failure.
    records = [
        _record("u1", "t1", ["t1", "x"], ["t1", "x"]),
        _record("u2", "t2", ["t2", "x"], ["x", "y"]),
        _record("u3", "t3", ["x", "y"], ["x", "y"]),
        _record("u4", "t4", ["t4", "x"], [], format_failure=True),
    ]
    result = evaluate_agentic(records, ks=(10,))
    assert result["candidate_target_coverage"] == pytest.approx(0.75)
    # In-pool users are u1, u2, u4; only u1 hits (u4's format failure empties its ranking).
    assert result["conditional_R@10_target_in_pool"] == pytest.approx(1.0 / 3.0)
    assert result["R@10"] == pytest.approx(0.75 * (1.0 / 3.0))


def test_coverage_factorization_identity_detects_tampering(monkeypatch):
    records = [
        _record("u1", "t1", ["t1", "x"], ["t1", "x"]),
        _record("u2", "t2", ["x", "y"], ["x", "y"]),
    ]
    import lime_rec.agentic.metrics as metrics_module
    original = metrics_module.evaluate_rankings
    monkeypatch.setattr(metrics_module, "evaluate_rankings",
                        lambda *a, **k: {**original(*a, **k), "R@10": 0.9})
    with pytest.raises(AssertionError, match="coverage factorization"):
        evaluate_agentic(records, ks=(10,))
