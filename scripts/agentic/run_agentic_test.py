"""Run frozen matched-evidence conditions on a fixed test candidate cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lime_rec.agentic.agent import ToolUsingAgent
from lime_rec.agentic.cache import LLMCache
from lime_rec.agentic.llm_client import (CachedLLMClient, OpenAICompatibleClient,
                                         RuleBasedMockLLMClient)
from lime_rec.agentic.metrics import evaluate_agentic
from lime_rec.agentic.protocol import (assert_same_candidate_pool,
                                       assert_same_expert_scores,
                                       assert_same_targets, assert_same_users)
from lime_rec.agentic.recovery import CandidateRecovery
from lime_rec.agentic.tools import ItemCFTool, SASRecTool, SemanticTool
from lime_rec.evaluation import ExpertEvaluator, load_configured_dataset
from lime_rec.protocols import evaluation_label


def _load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _record(base: dict, target: str, ranking: list[str], condition: str,
            format_failure: bool = False, tool_calls=None, retries=0, events=None) -> dict:
    rank = ranking.index(target) if target in ranking else None
    events = events or []
    return {"user_id": base["user_id"], "target_item_id": target,
            "candidate_ids": base["candidate_ids"], "candidate_hash": base["candidate_hash"],
            "expert_score_hashes": base["expert_score_hashes"], "ranking": ranking,
            "condition": condition, "target_rank": rank,
            "hit5": rank is not None and rank < 5, "hit10": rank is not None and rank < 10,
            "tool_calls": list(tool_calls or []), "format_failure": format_failure,
            "agent_failure": format_failure,
            "retry_count": retries, "llm_calls": len(events),
            "input_tokens": sum(e.get("usage", {}).get("prompt_tokens", 0) for e in events),
            "output_tokens": sum(e.get("usage", {}).get("completion_tokens", 0) for e in events),
            "latency_seconds": sum(e.get("latency_seconds", 0.0) for e in events),
            "cache_hit": bool(events) and all(e.get("cache_hit") for e in events)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sasrec-model", required=True)
    parser.add_argument("--itemcf-model", required=True)
    parser.add_argument("--semantic-emb", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llm-cache", default="outputs/agentic/llm_cache")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--json-schema", action="store_true", help="schema-constrained decoding variant; requires a fresh output/cache root")
    parser.add_argument("--resume", action="store_true",
                        help="Retained for explicit runbooks; cache reuse is the default.")
    parser.add_argument("--text-visible", action="store_true",
                        help="Expose candidate catalog text to the controller (variant protocol).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.json_schema and Path(args.out_dir).exists():
        raise FileExistsError("schema variant requires a new output directory")
    frozen = json.loads(Path(args.frozen_config).read_text())
    prompt_paths = ({"agentic": Path("prompts/agentic_v1_text.txt"),
                     "one_shot": Path("prompts/one_shot_v1_text.txt")}
                    if args.text_visible else
                    {"agentic": Path("prompts/agentic_v1.txt"),
                     "one_shot": Path("prompts/one_shot_v1.txt")})
    if args.text_visible:
        # 变体协议：冻结配置中的 prompt 哈希对应不透明版本，这里改校验变体文件存在性
        for path in prompt_paths.values():
            if not path.exists():
                raise RuntimeError(f"missing text-visible prompt: {path}")
    else:
        live_prompt_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in prompt_paths.items()}
        if live_prompt_hashes != frozen["prompt_sha256"]:
            raise RuntimeError("versioned prompt content differs from frozen test config")
    cache_meta = json.loads(Path(args.candidate_cache + ".metadata.json").read_text())
    if cache_meta["split"] != "test" or cache_meta["candidate_per_expert"] != frozen["candidate_per_expert"]:
        raise RuntimeError("test cache does not match frozen validation-selected configuration")
    cached_rows = _load_jsonl(args.candidate_cache)
    if args.max_users:
        cached_rows = cached_rows[:args.max_users]
    if args.dry_run:
        print(json.dumps({"validated_users": len(cached_rows), "network_calls": 0}))
        return
    dataset = load_configured_dataset(args.config)
    evaluator = ExpertEvaluator.from_paths(args.config, args.sasrec_model, args.itemcf_model,
                                           args.semantic_emb, device=args.device, mask_history=False)
    tools = {"sasrec": SASRecTool(evaluator), "itemcf": ItemCFTool(evaluator),
             "semantic": SemanticTool(evaluator)}
    checkpoint = torch.load(frozen["recovery_checkpoint"], map_location="cpu", weights_only=False)
    recovery = CandidateRecovery(device=args.device)
    recovery.gate.load_state_dict(checkpoint["state_dict"])
    recovery.fitted_split = checkpoint["fit_split"]
    agent_prompt = prompt_paths["agentic"].read_text()
    one_shot_prompt = prompt_paths["one_shot"].read_text()
    base_client = (RuleBasedMockLLMClient() if frozen["model"] == "rule-based-mock"
                   else OpenAICompatibleClient.from_environment())
    if args.json_schema:
        from scripts.revision.strict_controller import SchemaClient
        base_client = SchemaClient.from_environment()
    if base_client.model_name != frozen["model"]:
        raise RuntimeError("runtime LLM model differs from frozen model")
    suffix = "_text" if args.text_visible else ""
    item_texts = dataset.item_text if args.text_visible else None
    if args.text_visible and not item_texts:
        raise RuntimeError("text-visible variant requires dataset metadata with item text")
    outputs = {name: [] for name in ("recovery", "llm_all_tools", "adaptive_agent",
                                     "adaptive_agent_no_semantic", "forced_sequential")}
    for base in cached_rows:
        user_id, history, candidates = base["user_id"], base["history"], base["candidate_ids"]
        live_hashes = [tools[name].query(user_id, history, candidates)["score_hash"]
                       for name in ("sasrec", "itemcf", "semantic")]
        if live_hashes != base["expert_score_hashes"]:
            raise RuntimeError(f"expert-score cache mismatch for {user_id}")
        label = evaluation_label(dataset, user_id, "test")  # label is read only after context/cache exists
        scores = np.asarray(base["expert_scores"], dtype=np.float32)
        fused = recovery.score(scores[None], np.asarray(base["history_mask"])[None],
                               np.asarray([base["history_length"]]))[0]
        ranking = [candidates[i] for i in sorted(range(len(candidates)),
                   key=lambda i: (-float(fused[i]), candidates[i]))[:10]]
        outputs["recovery"].append(_record(base, label.target_item_id, ranking, "recovery"))
        for name, allowed, prompt, all_first, forced_seq in (
            ("llm_all_tools", tools, one_shot_prompt, True, None),
            ("adaptive_agent", tools, agent_prompt, False, None),
            ("adaptive_agent_no_semantic", {k: v for k, v in tools.items() if k != "semantic"}, agent_prompt, False, None),
            ("forced_sequential", tools, agent_prompt, False, ("sasrec", "itemcf", "semantic")),
        ):
            request_context = {"dataset": dataset.name, "split": "test", "user_id": user_id,
                 "candidate_hash": base["candidate_hash"], "prompt_version": prompt_paths[
                     "agentic" if prompt == agent_prompt else "one_shot"].name,
                 "agent_condition": name}
            if args.json_schema:
                from scripts.revision import strict_controller
                request_context["decoding_variant"] = "json-schema-anonymous"
                request_context["decoding_code_sha256"] = hashlib.sha256(
                    Path(strict_controller.__file__).read_bytes()).hexdigest()
            if args.text_visible:
                request_context["text_visible"] = True
            cached_client = CachedLLMClient(base_client, LLMCache(Path(args.llm_cache)),
                request_context, args.force_refresh)
            result = ToolUsingAgent(cached_client, allowed, prompt,
                frozen["max_tool_calls"], frozen["temperature"], frozen["max_tokens"],
                frozen["retry_count"], item_texts=item_texts).run(user_id, history, candidates,
                                                                  all_tools_first=all_first,
                                                                  forced_tool_sequence=forced_seq)
            state = result["state"]
            outputs[name].append(_record(base, label.target_item_id,
                state.final_ranking or [], name + suffix, result["format_failure"], state.called_tools,
                result["retries"], cached_client.events))
    protocol_view = {name: rows for name, rows in outputs.items() if name != "adaptive_agent_no_semantic"}
    assert_same_users(protocol_view); assert_same_targets(protocol_view)
    assert_same_candidate_pool(protocol_view); assert_same_expert_scores(protocol_view)
    out_dir = Path(args.out_dir)
    for name, rows in outputs.items():
        _write_jsonl(out_dir / f"{name}.jsonl", rows)
    summary = {name: evaluate_agentic(rows) for name, rows in outputs.items()}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
