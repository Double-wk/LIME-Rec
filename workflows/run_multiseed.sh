#!/usr/bin/env bash
# Train SASRec with multiple random seeds and evaluate a shared LIME-Rec gate.
#
# Example:
#   SEEDS="0 1 2" bash workflows/run_multiseed.sh
#
# Environment variables:
#   DATASETS="amazon_beauty amazon_toys amazon_sports"
#   SEEDS="0 1 2"
#   DEVICE=auto
#   ENCODER=bge_base
#   RUN_EVAL=1
#   SASREC_EVAL_EVERY=10
#   VALIDATION_SEED=0

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASETS="${DATASETS:-amazon_beauty amazon_toys amazon_sports}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-auto}"
ENCODER="${ENCODER:-bge_base}"
RUN_EVAL="${RUN_EVAL:-1}"
SHARED_WEIGHTS="${SHARED_WEIGHTS:-0.50,0.20,0.30}"
SASREC_EVAL_EVERY="${SASREC_EVAL_EVERY:-10}"
VALIDATION_SEED="${VALIDATION_SEED:-0}"

read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a SEED_LIST <<< "$SEEDS"
[ "${#DATASET_LIST[@]}" -gt 0 ] || { echo "[error] DATASETS is empty." >&2; exit 2; }
[ "${#SEED_LIST[@]}" -gt 0 ] || { echo "[error] SEEDS is empty." >&2; exit 2; }

REPORT_DIR="output/results/supplementary/multiseed"
mkdir -p logs outputs/models outputs/embeddings "$REPORT_DIR"

require_file() {
  local path="$1"
  [ -s "$path" ] || { echo "[error] required file is missing or empty: $path" >&2; exit 1; }
}

itemcf_is_valid() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

try:
    model = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
neighbors = model.get("neighbors") if isinstance(model, dict) else None
raise SystemExit(0 if isinstance(neighbors, dict) and neighbors else 1)
PY
}

ensure_itemcf() {
  local dataset="$1"
  local config="configs/${dataset}.json"
  local output="outputs/models/${dataset}_itemcf.json"
  require_file "$config"

  if itemcf_is_valid "$output"; then
    echo "[itemcf] $dataset: using $output"
    return
  fi

  echo "[itemcf] $dataset: model is missing or invalid; building it from training histories"
  "$PYTHON_BIN" -m scripts.training.build_itemcf \
    --config "$config" \
    --out "$output" \
    2>&1 | tee "logs/itemcf_${dataset}.log"
  itemcf_is_valid "$output" || {
    echo "[error] ItemCF build did not create a usable model: $output" >&2
    exit 1
  }
}

echo "=== TRAIN $(date) ==="
for seed in "${SEED_LIST[@]}"; do
  for dataset in "${DATASET_LIST[@]}"; do
    config="configs/${dataset}.json"
    output="outputs/models/${dataset}_sasrec_seed${seed}.pt"
    require_file "$config"
    if [ -s "$output" ]; then
      echo "[skip] $output exists"
      continue
    fi
    echo "[train] $dataset seed=$seed -> $output"
    "$PYTHON_BIN" -m scripts.training.train_sasrec \
      --config "$config" \
      --output "$output" \
      --device "$DEVICE" \
      --eval-every "$SASREC_EVAL_EVERY" \
      --seed "$seed" \
      --validation-seed "$VALIDATION_SEED" \
      > "logs/sasrec_${dataset}_seed${seed}.log" 2>&1
    echo "[done] $dataset seed=$seed $(date)"
  done
done

if [ "$RUN_EVAL" = "1" ]; then
  echo "=== EVAL $(date) ==="
  for dataset in "${DATASET_LIST[@]}"; do
    require_file "outputs/embeddings/${dataset}_${ENCODER}.npz"
    ensure_itemcf "$dataset"
  done

  datasets_csv="$(IFS=,; printf '%s' "${DATASET_LIST[*]}")"
  report_paths=()
  for seed in "${SEED_LIST[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
      require_file "outputs/models/${dataset}_sasrec_seed${seed}.pt"
    done
    echo "[eval] seed=$seed"
    "$PYTHON_BIN" -m scripts.evaluation.run_shared_weight \
      --weights "$SHARED_WEIGHTS" \
      --datasets "$datasets_csv" \
      --encoder "$ENCODER" \
      --sasrec-suffix "seed${seed}" \
      --out "${REPORT_DIR}/multiseed_shared_weight_${ENCODER}_seed${seed}.json" \
      > "logs/eval_${ENCODER}_seed${seed}.log" 2>&1
    report_paths+=("${REPORT_DIR}/multiseed_shared_weight_${ENCODER}_seed${seed}.json")
    echo "[done] evaluation seed=$seed $(date)"
  done
  if [ "${#report_paths[@]}" -ge 2 ]; then
    summary="${REPORT_DIR}/multiseed_shared_weight_${ENCODER}_summary.json"
    "$PYTHON_BIN" -m scripts.evaluation.summarize_multiseed \
      --inputs "${report_paths[@]}" \
      --out "$summary" \
      > "logs/eval_${ENCODER}_summary.log" 2>&1
    echo "[done] summary -> $summary $(date)"
  fi
else
  echo "=== EVAL SKIPPED (RUN_EVAL=0) ==="
fi

echo "=== ALL DONE $(date) ==="
