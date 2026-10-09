#!/usr/bin/env bash
# 单个 (domain, seed) 的选择器 LoRA 训练 + merge
# usage: bash scripts/agentic/run_train_selector.sh <domain> <train_seed> <gpu_id>
set -euo pipefail
cd "$(dirname "$0")/../.."

DOMAIN=$1
TRAIN_SEED=$2
GPU=$3
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
EXP="outputs/budgeted_tools_v1/amazon_${DOMAIN}/seed${TRAIN_SEED}"

test -f "$EXP/fitted/selector_sft.jsonl" || { echo "[$DOMAIN s$TRAIN_SEED] missing SFT data"; exit 1; }
test -f "$EXP/fitted/fit_manifest.json" || { echo "[$DOMAIN s$TRAIN_SEED] missing fit manifest"; exit 1; }
python - "$EXP/fitted/fit_manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m.get("sft_examples", 0) > 0, "fit_manifest sft_examples == 0"
print(f"[manifest] sft_examples={m['sft_examples']} skipped={m['sft_tied_states_skipped']}")
PY

test ! -e "$EXP/selector_adapter" && test ! -e "$EXP/selector_merged" || { echo "[$DOMAIN s$TRAIN_SEED] adapter/merged exists"; exit 1; }

echo "[$DOMAIN s$TRAIN_SEED] train LoRA (gpu $GPU)"
CUDA_VISIBLE_DEVICES=$GPU python -m scripts.agentic.train_adapted_controller \
  --model "$BASE_MODEL" --data "$EXP/fitted/selector_sft.jsonl" \
  --out "$EXP/selector_adapter" --max-examples 0 --epochs 1 --seed 2027

echo "[$DOMAIN s$TRAIN_SEED] merge adapter"
python -m scripts.agentic.merge_adapted_controller \
  --base "$BASE_MODEL" --adapter "$EXP/selector_adapter" --out "$EXP/selector_merged"

echo "[$DOMAIN s$TRAIN_SEED] === TRAIN+MERGE DONE ==="