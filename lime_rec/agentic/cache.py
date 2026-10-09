"""Secret-free content-addressed cache for LLM requests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def request_cache_key(request: Mapping) -> str:
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class LLMCache:
    root: Path

    def get(self, request: Mapping) -> dict | None:
        path = self.root / f"{request_cache_key(request)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, request: Mapping, raw_response: object, parsed_response: Mapping,
            latency_seconds: float, usage: Mapping | None = None) -> Path:
        forbidden = {"authorization", "api_key", "llm_api_key"}
        if forbidden & {str(key).lower() for key in request}:
            raise ValueError("cache request contains a secret-bearing field")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{request_cache_key(request)}.json"
        payload = {"request": dict(request), "raw_response": raw_response,
                   "parsed_response": dict(parsed_response),
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "latency_seconds": latency_seconds, "usage": dict(usage or {})}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path
