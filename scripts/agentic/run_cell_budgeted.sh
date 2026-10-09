#!/usr/bin/env bash
# 幂等续跑单个 (domain, seed) 格子：只执行缺失阶段，已有产物不覆盖。
# usage: bash scripts/agentic/run_cell_budgeted.sh <domain> <train_seed> <gpu_id>
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
test -f "$SAS" && test -f "$CF" && test -f "$EMB" || { echo "[$DOMAIN s$TRAIN_SEED] missing artifacts"; exit 1; }
mkdir -p "$EXP/cache"

# 半成品 cache（只有 jsonl 或只有 metadata）视为失败残留，清理后重建
for split in validation test; do
  J="$EXP/cache/$split.jsonl"; M="$J.metadata.json"
  if [ -e "$J" ] || [ -e "$M" ]; then
    if [ ! -e "$J" ] || [ ! -e "$M" ]; then
      echo "[$DOMAIN s$TRAIN_SEED] removing incomplete $split cache"
      rm -f "$J" "$M"
    fi
  fi
done

if [ ! -e "$EXP/cache/validation.jsonl" ]; then
  echo "[$DOMAIN s$TRAIN_SEED] build validation cache (gpu $GPU)"
  python -m scripts.agentic.build_candidate_cache \
    --config "configs/amazon_${DOMAIN}.json" \
    --sasrec-model "$SAS" --itemcf-model "$CF" --semantic-emb "$EMB" \
    --split validation --candidate-per-expert 20 --sample-seed 2027 \
    --device cuda --out "$EXP/cache/validation.jsonl"
else
  echo "[$DOMAIN s$TRAIN_SEED] validation cache exists, skip build"
fi

if [ ! -e "$EXP/cache/test.jsonl" ]; then
  echo "[$DOMAIN s$TRAIN_SEED] build test cache $TEST_USERS users (gpu $GPU)"
  python -m scripts.agentic.build_candidate_cache \
    --config "configs/amazon_${DOMAIN}.json" \
    --sasrec-model "$SAS" --itemcf-model "$CF" --semantic-emb "$EMB" \
    --split test --candidate-per-expert 20 --sample-seed 2027 \
    --device cuda --max-users "$TEST_USERS" --out "$EXP/cache/test.jsonl"
else
  echo "[$DOMAIN s$TRAIN_SEED] test cache exists, skip build"
fi

if [ ! -e "$EXP/validation.json" ]; then
  echo "[$DOMAIN s$TRAIN_SEED] prepare validation (gpu $GPU)"
  python -m scripts.agentic.run_budgeted_tools prepare \
    --config "configs/amazon_${DOMAIN}.json" \
    --candidate-cache "$EXP/cache/validation.jsonl" --split validation \
    --device cuda --out "$EXP/validation.json"
else
  echo "[$DOMAIN s$TRAIN_SEED] validation.json exists, skip prepare"
fi

if [ ! -e "$EXP/test.json" ]; then
  echo "[$DOMAIN s$TRAIN_SEED] prepare test (gpu $GPU)"
  python -m scripts.agentic.run_budgeted_tools prepare \
    --config "configs/amazon_${DOMAIN}.json" \
    --candidate-cache "$EXP/cache/test.jsonl" --split test \
    --device cuda --out "$EXP/test.json"
else
  echo "[$DOMAIN s$TRAIN_SEED] test.json exists, skip prepare"
fi

if [ ! -e "$EXP/fitted" ]; then
  echo "[$DOMAIN s$TRAIN_SEED] fit router"
  python -m scripts.agentic.run_budgeted_tools fit \
    --data "$EXP/validation.json" --out-dir "$EXP/fitted"
else
  echo "[$DOMAIN s$TRAIN_SEED] fitted exists, skip fit"
fi

if [ ! -e "$EXP/baselines" ]; then
  echo "[$DOMAIN s$TRAIN_SEED] evaluate baselines"
  python -m scripts.agentic.run_budgeted_tools evaluate \
    --data "$EXP/test.json" --router "$EXP/fitted/router.json" \
    --out-dir "$EXP/baselines"
else
  echo "[$DOMAIN s$TRAIN_SEED] baselines exist, skip evaluate"
fi

echo "[$DOMAIN s$TRAIN_SEED] === CELL DONE ==="