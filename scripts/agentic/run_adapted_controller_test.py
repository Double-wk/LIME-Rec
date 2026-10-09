"""Evaluate the task-adapted controller under the frozen llm_all_tools protocol.

Runs exactly one condition: llm_all_tools with the served adapted model
(LoRA-merged checkpoint behind an OpenAI-compatible endpoint). Uses the same
frozen test candidate cache, prompt file, temperature, and token budget as the
prompt-only conditions; writes `llm_all_tools_adapted.jsonl` and metrics into
the agentic result directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lime_rec.agentic.agent import ToolUsingAgent
from lime_rec.agentic.cache import LLMCache
from lime_rec.agentic.llm_client import CachedLLMClient, OpenAICompatibleClient
from lime_rec.agentic.metrics import evaluate_agentic
from lime_rec.agentic.tools import ItemCFTool, SASRecTool, SemanticTool
from lime_rec.evaluation import ExpertEvaluator, load_configured_dataset
from lime_rec.protocols import evaluation_label


def _load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sasrec-model", required=True)
    parser.add_argument("--itemcf-model", required=True)
    parser.add_argument("--semantic-emb", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--frozen-config", required=True,
                        help="Frozen config; 'model' field must match the served model name")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llm-cache", default="outputs/agentic/llm_cache_adapted")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    frozen = json.loads(Path(args.frozen_config).read_text())
    prompt = Path("prompts/one_shot_v1.txt").read_text()
    cache_meta = json.loads(Path(args.candidate_cache + ".metadata.json").read_text())
    if cache_meta["split"] != "test":
        raise RuntimeError("adapted evaluation must run on the frozen test cache")
    cached_rows = _load_jsonl(args.candidate_cache)

    dataset = load_configured_dataset(args.config)
    evaluator = ExpertEvaluator.from_paths(args.config, args.sasrec_model, args.itemcf_model,
                                           args.semantic_emb, device=args.device, mask_history=False)
    tools = {"sasrec": SASRecTool(evaluator), "itemcf": ItemCFTool(evaluator),
             "semantic": SemanticTool(evaluator)}

    base_client = OpenAICompatibleClient.from_environment()
    if base_client.model_name != frozen["model"]:
        raise RuntimeError(f"runtime model {base_client.model_name} != frozen {frozen['model']}")

    rows = []
    for base in cached_rows:
        user_id, history, candidates = base["user_id"], base["history"], base["candidate_ids"]
        live_hashes = [tools[name].query(user_id, history, candidates)["score_hash"]
                       for name in ("sasrec", "itemcf", "semantic")]
        if live_hashes != base["expert_score_hashes"]:
            raise RuntimeError(f"expert-score cache mismatch for {user_id}")
        label = evaluation_label(dataset, user_id, "test")
        request_context = {"dataset": dataset.name, "split": "test", "user_id": user_id,
                           "candidate_hash": base["candidate_hash"],
                           "prompt_version": "one_shot_v1", "agent_condition": "llm_all_tools_adapted"}
        client = CachedLLMClient(base_client, LLMCache(Path(args.llm_cache)), request_context, False)
        result = ToolUsingAgent(client, tools, prompt,
                                frozen["max_tool_calls"], frozen["temperature"],
                                frozen["max_tokens"], frozen["retry_count"]).run(
            user_id, history, candidates, all_tools_first=True)
        state = result["state"]
        ranking = state.final_ranking or []
        target = label.target_item_id
        rank = ranking.index(target) if target in ranking else None
        rows.append({"user_id": user_id, "target_item_id": target,
                     "candidate_ids": candidates, "candidate_hash": base["candidate_hash"],
                     "expert_score_hashes": base["expert_score_hashes"], "ranking": ranking,
                     "condition": "llm_all_tools_adapted", "target_rank": rank,
                     "hit5": rank is not None and rank < 5,
                     "hit10": rank is not None and rank < 10,
                     "tool_calls": list(state.called_tools),
                     "format_failure": result["format_failure"],
                     "agent_failure": result["format_failure"],
                     "retry_count": result["retries"],
                     "llm_calls": len(client.events),
                     "input_tokens": sum(e.get("usage", {}).get("prompt_tokens", 0) for e in client.events),
                     "output_tokens": sum(e.get("usage", {}).get("completion_tokens", 0) for e in client.events),
                     "latency_seconds": sum(e.get("latency_seconds", 0.0) for e in client.events),
                     "cache_hit": bool(client.events) and all(e.get("cache_hit") for e in client.events)})
        if len(rows) % 100 == 0:
            print(f"[adapted] {len(rows)}/{len(cached_rows)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "llm_all_tools_adapted.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metrics = evaluate_agentic(rows)
    (out_dir / "metrics_adapted.json").write_text(json.dumps(
        {"llm_all_tools_adapted": metrics}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
