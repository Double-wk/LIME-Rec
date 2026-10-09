import json
from pathlib import Path

from scripts.baselines.audit_gram_alignment import audit


def test_gram_alignment_fixture(tmp_path, monkeypatch):
    interactions = tmp_path / "train.jsonl"
    rows = [{"user_id": "u", "item_id": item, "timestamp": i}
            for i, item in enumerate(["a", "b", "c", "d", "e"], 1)]
    interactions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"name": "toy", "interactions_path": str(interactions),
                                  "min_user_interactions": 1, "min_item_interactions": 1}))
    sequence = tmp_path / "user_sequence.txt"
    sequence.write_text("u a b c d e\n")
    monkeypatch.setattr("scripts.baselines.audit_gram_alignment.subprocess.run",
                        lambda *a, **k: type("R", (), {"stdout": "abc\n"})())
    result = audit(str(config), str(sequence), str(tmp_path))
    assert result["passed"] is True
    assert result["target_mismatches"] == 0
