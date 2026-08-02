import json

import numpy as np
import pytest

from lime_rec.experts import ItemCFExpert
from lime_rec.itemcf import build_itemcf_model


def test_itemcf_builder_is_deterministic_and_scores_catalog_items(tmp_path):
    model = build_itemcf_model(
        dataset_name="toy",
        histories={"u1": ["a", "b", "c"], "u2": ["a", "b"]},
        item_ids=["a", "b", "c", "d"],
        top_k=1,
        window_size=2,
    )

    assert model["training_only"] is True
    assert model["num_training_interactions"] == 5
    assert model["neighbors"]["a"] == [{"item_id": "b", "score": 1.0}]
    assert model["neighbors"]["b"] == [{"item_id": "a", "score": 1.0}]

    path = tmp_path / "itemcf.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    expert = ItemCFExpert(str(path), ["a", "b", "c", "d"])

    scores = expert.score("u3", ["a"])
    assert scores.shape == (4,)
    assert scores[1] == pytest.approx(1.0)
    assert np.count_nonzero(scores) == 1


def test_itemcf_rejects_an_empty_model(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"neighbors": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="contains no neighbours"):
        ItemCFExpert(str(path), ["a", "b"])
