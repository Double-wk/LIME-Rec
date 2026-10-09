"""Bounded single-agent loop over fixed recommendation tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from .protocol import AgentState, validate_action


def parse_action(text: str) -> dict:
    try:
        action = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response is not valid JSON") from exc
    if not isinstance(action, dict):
        raise ValueError("LLM action must be a JSON object")
    return action


@dataclass
class ToolUsingAgent:
    client: object
    tools: Mapping[str, object]
    prompt: str
    max_tool_calls: int = 3
    temperature: float = 0.0
    max_tokens: int = 512
    retry_count: int = 2
    # text-visible 变体：非 None 时在消息中附带候选 item 文本（对照协议默认不透明 ID）
    item_texts: Mapping[str, str] | None = None

    def run(self, user_id: str, history: list[str], candidates: list[str],
            *, all_tools_first: bool = False,
            forced_tool_sequence: Sequence[str] | None = None) -> dict:
        state = AgentState.create(user_id, history, candidates, self.max_tool_calls)
        if forced_tool_sequence:
            # 强制顺序观察：控制器无权选择工具，逐个注入 call_tool 决策与观测后再排序。
            # 与 all_tools_first 的差异：观测以多轮对话形式逐步呈现，而非一次性并列。
            for name in forced_tool_sequence:
                tool = self.tools[name]
                state.messages.append(self._message(state, tuple(self.tools), force_finish=False))
                call = {"action": "call_tool", "tool": name}
                state.messages.append({"role": "assistant", "content": json.dumps(call)})
                state.observations[name] = tool.query(user_id, history, candidates)
                state.called_tools.append(name)
                state.remaining_budget -= 1
        elif all_tools_first:
            for name, tool in self.tools.items():
                state.observations[name] = tool.query(user_id, history, candidates)
                state.called_tools.append(name)
            state.remaining_budget = 0
        failures = 0
        while True:
            message = self._message(state, tuple(self.tools), force_finish=state.remaining_budget == 0)
            state.messages.append(message)
            try:
                response = self.client.generate([{"role": "system", "content": self.prompt}, message],
                                                temperature=self.temperature, max_tokens=self.max_tokens)
                action = parse_action(response["text"])
                state.messages.append({"role": "assistant", "content": response["text"]})
                validate_action(action, state, tuple(self.tools))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                failures += 1
                if failures > self.retry_count:
                    return {"state": state, "format_failure": True,
                            "agent_failure": True, "retries": failures - 1}
                continue
            if action["action"] == "finish":
                state.final_ranking = list(action["ranking"])
                return {"state": state, "format_failure": False,
                        "agent_failure": False, "retries": failures}
            name = action["tool"]
            state.observations[name] = self.tools[name].query(user_id, history, candidates)
            state.called_tools.append(name)
            state.remaining_budget -= 1

    def _message(self, state: AgentState, available_tools: tuple[str, ...], force_finish: bool) -> dict:
        payload = {"history_length": state.history_length,
                   "candidate_ids": state.candidate_ids,
                   "is_history_item": state.history_membership,
                   "available_tools": available_tools, "called_tools": state.called_tools,
                   "remaining_tool_budget": state.remaining_budget,
                   "observations": state.observations}
        if self.item_texts is not None:
            payload["candidate_text"] = {item: (self.item_texts.get(item, "") or "")[:200]
                                         for item in state.candidate_ids}
        instruction = "You must now return final ranking." if force_finish else "Return one JSON action."
        return {"role": "user", "content": instruction + "\n" + json.dumps(payload)}
