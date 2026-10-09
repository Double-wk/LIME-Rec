"""Vendor-neutral generation interface and OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import requests

from .cache import LLMCache


class LLMClient(Protocol):
    model_name: str

    def generate(self, messages: Sequence[dict], *, temperature: float, max_tokens: int) -> dict: ...


@dataclass
class MockLLMClient:
    responses: list[dict]
    model_name: str = "mock"

    def generate(self, messages: Sequence[dict], *, temperature: float = 0.0,
                 max_tokens: int = 512) -> dict:
        if not self.responses:
            raise RuntimeError("MockLLMClient has no remaining responses")
        return {"text": json.dumps(self.responses.pop(0)), "usage": {}, "raw": None}


@dataclass
class RuleBasedMockLLMClient:
    """Offline controller that exercises the complete bounded agent loop."""

    model_name: str = "rule-based-mock"

    def generate(self, messages: Sequence[dict], *, temperature: float = 0.0,
                 max_tokens: int = 512) -> dict:
        payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
        called = payload["called_tools"]
        remaining = payload["remaining_tool_budget"]
        for name in payload["available_tools"]:
            if remaining and name not in called:
                return {"text": json.dumps({"action": "call_tool", "tool": name}),
                        "usage": {}, "raw": None}
        candidates = payload["candidate_ids"]
        observations = payload["observations"]
        totals = {item: 0.0 for item in candidates}
        for observation in observations.values():
            for row in observation["items"]:
                totals[row["item_id"]] += float(row["score"])
        ranking = sorted(candidates, key=lambda item: (-totals[item], item))[:10]
        return {"text": json.dumps({"action": "finish", "ranking": ranking}),
                "usage": {}, "raw": None}


@dataclass
class OpenAICompatibleClient:
    base_url: str
    api_key: str
    model_name: str
    timeout: float = 60.0

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleClient":
        required = {name: os.environ.get(name) for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")}
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"missing LLM environment variables: {', '.join(missing)}")
        return cls(required["LLM_BASE_URL"], required["LLM_API_KEY"], required["LLM_MODEL"])

    def generate(self, messages: Sequence[dict], *, temperature: float, max_tokens: int) -> dict:
        response = requests.post(
            self.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model_name, "messages": list(messages),
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return {"text": payload["choices"][0]["message"]["content"],
                "usage": payload.get("usage", {}), "raw": payload}


@dataclass
class CachedLLMClient:
    client: LLMClient
    cache: LLMCache
    request_context: dict
    force_refresh: bool = False

    def __post_init__(self):
        self.model_name = self.client.model_name
        self.events: list[dict] = []

    def generate(self, messages: Sequence[dict], *, temperature: float, max_tokens: int) -> dict:
        request = {**self.request_context, "model_name": self.model_name,
                   "temperature": temperature, "max_tokens": max_tokens,
                   "messages": list(messages)}
        cached = None if self.force_refresh else self.cache.get(request)
        if cached is not None:
            self.events.append({"cache_hit": True, "latency_seconds": 0.0,
                                "usage": cached.get("usage", {})})
            return {"text": cached["raw_response"], "usage": cached.get("usage", {}),
                    "raw": None}
        started = time.monotonic()
        response = self.client.generate(messages, temperature=temperature, max_tokens=max_tokens)
        latency = time.monotonic() - started
        self.cache.put(request, response["text"], {}, latency, response.get("usage"))
        self.events.append({"cache_hit": False, "latency_seconds": latency,
                            "usage": response.get("usage", {})})
        return response
