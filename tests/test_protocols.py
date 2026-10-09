import pytest

from lime_rec.data import Dataset
from lime_rec.protocols import evaluation_label, prediction_context, truncate_history


def _dataset():
    return Dataset("toy", ["u"], ["a", "b", "c", "d"], {"u": ["a", "b"]},
                   {"u": "c"}, {"u": "d"}, {})


def test_history_truncation_is_last_n():
    assert truncate_history(["a", "b", "c"], 2) == ["b", "c"]
    assert truncate_history(["a"], None) == ["a"]
    with pytest.raises(ValueError):
        truncate_history(["a"], 0)


def test_test_history_contains_validation_item_before_truncation():
    context = prediction_context(_dataset(), "u", "test")
    assert context.history == ("a", "b", "c")
    assert evaluation_label(_dataset(), "u", "test").target_item_id == "d"


def test_test_history_contains_validation_item_before_last_20_truncation():
    history = [f"i{i}" for i in range(21)]
    dataset = Dataset("toy", ["u"], history + ["valid", "test"], {"u": history},
                      {"u": "valid"}, {"u": "test"}, {})
    controlled = truncate_history(prediction_context(dataset, "u", "test").history, 20)
    assert controlled[-1] == "valid"
    assert controlled[0] == "i2"
    assert "test" not in controlled
