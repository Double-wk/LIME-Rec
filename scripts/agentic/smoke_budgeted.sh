#!/usr/bin/env bash
# Smoke 验收：beauty seed0，test 50 用户，验证 budgeted_tools 全流程
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN_ROOT=outputs/budgeted_tools_smoke
DOMAIN=beauty
TRAIN_SEED=0
TEST_USERS=50
EXP="$RUN_ROOT/amazon_${DOMAIN}/seed${TRAIN_SEED}"

export CUDA_VISIBLE_DEVICES=0

SAS="outputs/models/controlled_gram/amazon_${DOMAIN}_sasrec_max20_seed${TRAIN_SEED}.pt"
CF="output_final/models/amazon_${DOMAIN}_itemcf.json"
EMB="output_final/models/amazon_${DOMAIN}_bge_base.npz"

test -f "$SAS" && test -f "$CF" && test -f "$EMB" || { echo "model files missing"; exit 1; }
test ! -e "$EXP" || { echo "Output exists: $EXP"; exit 1; }
mkdir -p "$EXP/cache"

echo "=== build validation cache ==="
python -m scripts.agentic.build_candidate_cache \
  --config "configs/amazon_${DOMAIN}.json" \
  --sasrec-model "$SAS" --itemcf-model "$CF" --semantic-emb "$EMB" \
  --split validation --candidate-per-expert 20 --sample-seed 2027 \
  --device cuda --out "$EXP/cache/validation.jsonl"

echo "=== build test cache ($TEST_USERS users) ==="
python -m scripts.agentic.build_candidate_cache \
  --config "configs/amazon_${DOMAIN}.json" \
  --sasrec-model "$SAS" --itemcf-model "$CF" --semantic-emb "$EMB" \
  --split test --candidate-per-expert 20 --sample-seed 2027 \
  --device cuda --max-users "$TEST_USERS" --out "$EXP/cache/test.jsonl"

echo "=== prepare validation ==="
python -m scripts.agentic.run_budgeted_tools prepare \
  --config "configs/amazon_${DOMAIN}.json" \
  --candidate-cache "$EXP/cache/validation.jsonl" --split validation \
  --device cuda --out "$EXP/validation.json"

echo "=== prepare test ==="
python -m scripts.agentic.run_budgeted_tools prepare \
  --config "configs/amazon_${DOMAIN}.json" \
  --candidate-cache "$EXP/cache/test.jsonl" --split test \
  --device cuda --out "$EXP/test.json"

echo "=== fit ==="
python -m scripts.agentic.run_budgeted_tools fit \
  --data "$EXP/validation.json" --out-dir "$EXP/fitted"

echo "=== evaluate baselines ==="
python -m scripts.agentic.run_budgeted_tools evaluate \
  --data "$EXP/test.json" --router "$EXP/fitted/router.json" \
  --out-dir "$EXP/baselines"

echo "=== DONE ==="
ls -R "$EXP" | head -40