import pytest

from lime_rec.agentic.agent import ToolUsingAgent
from lime_rec.agentic.llm_client import MockLLMClient
from lime_rec.agentic.metrics import evaluate_agentic
from lime_rec.agentic.protocol import AgentState, validate_action


def _state(budget=3):
    return AgentState.create("u", ["a"], list("abcdefghij"), budget)


def test_agent_cannot_call_unknown_tool():
    with pytest.raises(ValueError, match="unknown"):
        validate_action({"action": "call_tool", "tool": "web"}, _state(), ["sasrec"])


def test_agent_cannot_repeat_tool():
    state = _state()
    state.called_tools.append("sasrec")
    with pytest.raises(ValueError, match="repeat"):
        validate_action({"action": "call_tool", "tool": "sasrec"}, state, ["sasrec"])


def test_agent_budget_is_enforced():
    with pytest.raises(ValueError, match="exhausted"):
        validate_action({"action": "call_tool", "tool": "sasrec"}, _state(0), ["sasrec"])


def test_agent_ranking_must_be_subset_of_candidates():
    with pytest.raises(ValueError, match="subset"):
        validate_action({"action": "finish", "ranking": list("abcdefghi") + ["x"]},
                        _state(), ["sasrec"])


def test_invalid_agent_user_is_not_dropped():
    client = MockLLMClient([{"action": "bad"}])
    result = ToolUsingAgent(client, {}, "prompt", max_tool_calls=0, retry_count=0).run(
        "u", ["a"], list("abcdefghij"))
    assert result["format_failure"] is True
    metrics = evaluate_agentic([{"user_id": "u", "target_item_id": "a",
                                 "candidate_ids": list("abcdefghij"), "ranking": [],
                                 "format_failure": True}])
    assert metrics["R@10"] == 0.0
    assert metrics["format_failure_rate"] == 1.0
