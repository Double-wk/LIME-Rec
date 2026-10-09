#!/usr/bin/env bash
# 单个 (domain, train_seed) 的正式 budgeted_tools 组合：build caches → prepare → fit → evaluate
# usage: bash scripts/agentic/run_combo_budgeted.sh <domain> <train_seed> <gpu_id>
set -euo pipefail
cd "$(dirname "$0")/../.."

DOMAIN=$1
TRAIN_SEED=$2
GPU=$3
TEST_USERS=1000
RUN_ROOT=outputs/budgeted_tools_v1
EXP="$RUN_ROOT/amazon_${DOMAIN}/seed${TRAIN_SEED}"

export CUDA_VISIBLE_DEVICES=$GPU

SAS="outputs/models/controlled_gram/amazon_${DOMAIN}_sasrec_max20_seed${TRAIN_SEED}.pt"
CF="output_final/models/amazon_${DOMAIN}_itemcf.json"
EMB="output_final/models/amazon_${DOMAIN}_bge_base.npz"

test -f "$SAS" && test -f "$CF" && test -f "$EMB" || { echo "[$DOMAIN s$TRAIN_SEED] missing model artifacts"; exit 1; }
test ! -e "$EXP" || { echo "[$DOMAIN s$TRAIN_SEED] output exists: $EXP"; exit 1; }
mkdir -p "$EXP/cache"

echo "[$DOMAIN s$TRAIN_SEED] build validation cache (gpu $GPU)"
python -m scripts.agentic.build_candidate_cache \
  --config "configs/amazon_${DOMAIN}.json" \
  --sasrec-model "$SAS" --itemcf-model "$CF" --semantic-emb "$EMB" \
  --split validation --candidate-per-expert 20 --sample-seed 2027 \
  --device cuda --out "$EXP/cache/validation.jsonl"

echo "[$DOMAIN s$TRAIN_SEED] build test cache $TEST_USERS users (gpu $GPU)"
python -m scripts.agentic.build_candidate_cache \
  --config "configs/amazon_${DOMAIN}.json" \
  --sasrec-model "$SAS" --itemcf-model "$CF" --semantic-emb "$EMB" \
  --split test --candidate-per-expert 20 --sample-seed 2027 \
  --device cuda --max-users "$TEST_USERS" --out "$EXP/cache/test.jsonl"

echo "[$DOMAIN s$TRAIN_SEED] prepare validation (gpu $GPU)"
python -m scripts.agentic.run_budgeted_tools prepare \
  --config "configs/amazon_${DOMAIN}.json" \
  --candidate-cache "$EXP/cache/validation.jsonl" --split validation \
  --device cuda --out "$EXP/validation.json"

echo "[$DOMAIN s$TRAIN_SEED] prepare test (gpu $GPU)"
python -m scripts.agentic.run_budgeted_tools prepare \
  --config "configs/amazon_${DOMAIN}.json" \
  --candidate-cache "$EXP/cache/test.jsonl" --split test \
  --device cuda --out "$EXP/test.json"

echo "[$DOMAIN s$TRAIN_SEED] fit router"
python -m scripts.agentic.run_budgeted_tools fit \
  --data "$EXP/validation.json" --out-dir "$EXP/fitted"

echo "[$DOMAIN s$TRAIN_SEED] evaluate baselines"
python -m scripts.agentic.run_budgeted_tools evaluate \
  --data "$EXP/test.json" --router "$EXP/fitted/router.json" \
  --out-dir "$EXP/baselines"

echo "[$DOMAIN s$TRAIN_SEED] === COMBO DONE ==="