import json

import numpy as np

from lime_rec.agentic.tools import ItemCFTool
from lime_rec.data import Dataset
from lime_rec.evaluation import ExpertEvaluator
from lime_rec.experts import ItemCFExpert, LLMSemanticExpert


class FakeSAS:
    item_index = {"a": 0, "b": 1}
    def score(self, history): return np.array([0.2, 0.4], dtype=np.float32)


def test_tool_scores_equal_expert_evaluator(tmp_path):
    model = {"scoring_history_length": 20,
             "neighbors": {"a": [{"item_id": "b", "score": 2.0}]}}
    path = tmp_path / "itemcf.json"
    path.write_text(json.dumps(model))
    itemcf = ItemCFExpert(str(path), ["a", "b"])
    semantic = LLMSemanticExpert(["a", "b"], {}, embeddings=np.eye(2, dtype=np.float32))
    evaluator = ExpertEvaluator(Dataset("d", ["u"], ["a", "b"], {"u": ["a"]},
                                        {"u": "b"}, {"u": "b"}, {}),
                                FakeSAS(), itemcf, semantic, mask_history=False)
    output = ItemCFTool(evaluator).query("u", ["a"], ["a", "b"])
    expected = evaluator.normalized_scores("u", ["a"])[1]
    np.testing.assert_allclose([row["score"] for row in output["items"]], expected)
