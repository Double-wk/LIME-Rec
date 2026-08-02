#!/usr/bin/env bash
# Run the public LIME-Rec pipeline from raw benchmark data to evaluation.
#
# Example:
#   bash workflows/run_full_pipeline.sh
#
# Environment variables:
#   DEVICE=auto                         cpu, cuda, mps, or auto (default: auto)
#   DATASETS="amazon_beauty amazon_toys amazon_sports"
#   SEED=0
#   SKIP_DOWNLOAD=0 SKIP_ITEMCF=0 SKIP_EMBED=0 SKIP_TRAIN=0 SKIP_EVAL=0
#   ENCODER=bge_base
#   EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
#   SHARED_WEIGHTS=0.50,0.20,0.30
#   RUN_GATE_SELECTION=0 RUN_PAIRWISE_ABLATION=0
#     (pairwise ablation automatically runs gate selection first)
#
# Generated datasets, models, embeddings, logs, and evaluation outputs are
# written under data/, outputs/, and logs/. They are intentionally not tracked.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
DATASETS="${DATASETS:-amazon_beauty amazon_toys amazon_sports}"
SEED="${SEED:-0}"
ENCODER="${ENCODER:-bge_base}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-base-en-v1.5}"
SHARED_WEIGHTS="${SHARED_WEIGHTS:-0.50,0.20,0.30}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_ITEMCF="${SKIP_ITEMCF:-0}"
SKIP_EMBED="${SKIP_EMBED:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RUN_GATE_SELECTION="${RUN_GATE_SELECTION:-0}"
RUN_PAIRWISE_ABLATION="${RUN_PAIRWISE_ABLATION:-0}"

# Pairwise ablations use the validation-selected weights, so always create
# those weights first when ablations are requested.
if [ "$RUN_PAIRWISE_ABLATION" = "1" ]; then
  RUN_GATE_SELECTION=1
fi

SASREC_EVAL_EVERY="${SASREC_EVAL_EVERY:-10}"

read -r -a DATASET_LIST <<< "$DATASETS"
if [ "${#DATASET_LIST[@]}" -eq 0 ]; then
  echo "[error] DATASETS must contain at least one configured dataset." >&2
  exit 2
fi

mkdir -p data logs outputs/models outputs/embeddings

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [ -s "$path" ] || die "Required file is missing or empty: $path"
}

if [ "$RUN_PAIRWISE_ABLATION" = "1" ] && [ "$RUN_GATE_SELECTION" != "1" ]; then
  log "Pairwise ablation requires validation-selected gates; enabling gate selection first."
  RUN_GATE_SELECTION=1
fi

dataset_config() {
  local dataset="$1"
  local config="configs/${dataset}.json"
  require_file "$config"
  printf '%s\n' "$config"
}

download_name() {
  case "$1" in
    amazon_beauty) printf 'beauty\n' ;;
    amazon_toys) printf 'toys\n' ;;
    amazon_sports) printf 'sports\n' ;;
    yelp) printf 'yelp\n' ;;
    *) return 1 ;;
  esac
}

data_paths_present() {
  local dataset="$1"
  local config
  config="$(dataset_config "$dataset")"
  "$PYTHON_BIN" - "$config" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
paths = [config["interactions_path"], config.get("metadata_path", "")]
raise SystemExit(0 if all(path and Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths) else 1)
PY
}

step_download() {
  if [ "$SKIP_DOWNLOAD" = "1" ]; then
    log "Skipping download (SKIP_DOWNLOAD=1)."
    return
  fi

  log "Step 1/6: download benchmark data"
  local dataset source
  for dataset in "${DATASET_LIST[@]}"; do
    if data_paths_present "$dataset"; then
      log "[$dataset] data files already exist; skipping download."
      continue
    fi
    source="$(download_name "$dataset")" || die "No automatic downloader is available for '$dataset'. Provide its configured data files and rerun with SKIP_DOWNLOAD=1."
    log "[$dataset] downloading GRAM data"
    "$PYTHON_BIN" -m scripts.data.download_gram_data "$source" 2>&1 | tee "logs/download_${dataset}.log"
    data_paths_present "$dataset" || die "[$dataset] downloader did not create the configured interaction and metadata files."
  done
}

step_build_itemcf() {
  if [ "$SKIP_ITEMCF" = "1" ]; then
    log "Skipping ItemCF construction (SKIP_ITEMCF=1)."
    return
  fi

  log "Step 2/6: build training-only ItemCF models"
  local dataset config output
  for dataset in "${DATASET_LIST[@]}"; do
    config="$(dataset_config "$dataset")"
    output="outputs/models/${dataset}_itemcf.json"
    log "[$dataset] building $output"
    "$PYTHON_BIN" -m scripts.training.build_itemcf \
      --config "$config" \
      --out "$output" \
      2>&1 | tee "logs/itemcf_${dataset}.log"
    require_file "$output"
  done
}

step_build_embeddings() {
  if [ "$SKIP_EMBED" = "1" ]; then
    log "Skipping semantic embedding construction (SKIP_EMBED=1)."
    return
  fi

  log "Step 3/6: build semantic item embeddings"
  local dataset config output
  for dataset in "${DATASET_LIST[@]}"; do
    config="$(dataset_config "$dataset")"
    output="outputs/embeddings/${dataset}_${ENCODER}.npz"
    if [ -s "$output" ]; then
      log "[$dataset] embedding already exists; skipping $output"
      continue
    fi
    log "[$dataset] building $output with $EMBEDDING_MODEL"
    "$PYTHON_BIN" -m scripts.training.build_item_embeddings \
      --config "$config" \
      --model "$EMBEDDING_MODEL" \
      --device "$DEVICE" \
      --out "$output" \
      2>&1 | tee "logs/embedding_${dataset}_${ENCODER}.log"
    require_file "$output"
  done
}

step_train_sasrec() {
  if [ "$SKIP_TRAIN" = "1" ]; then
    log "Skipping SASRec training (SKIP_TRAIN=1)."
    return
  fi

  log "Step 4/6: train SASRec models"
  local dataset config output
  for dataset in "${DATASET_LIST[@]}"; do
    config="$(dataset_config "$dataset")"
    output="outputs/models/${dataset}_sasrec.pt"
    if [ -s "$output" ]; then
      log "[$dataset] checkpoint already exists; skipping $output"
      continue
    fi
    log "[$dataset] training $output (seed=$SEED)"
    "$PYTHON_BIN" -m scripts.training.train_sasrec \
      --config "$config" \
      --output "$output" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --eval-every "$SASREC_EVAL_EVERY" \
      2>&1 | tee "logs/sasrec_${dataset}_seed${SEED}.log"
    require_file "$output"
  done
}

check_evaluation_inputs() {
  local dataset
  for dataset in "${DATASET_LIST[@]}"; do
    require_file "outputs/models/${dataset}_itemcf.json"
    require_file "outputs/embeddings/${dataset}_${ENCODER}.npz"
    require_file "outputs/models/${dataset}_sasrec.pt"
  done
}

step_evaluate() {
  if [ "$SKIP_EVAL" = "1" ]; then
    log "Skipping evaluation (SKIP_EVAL=1)."
    return
  fi

  check_evaluation_inputs
  local datasets_csv dataset_tag
  datasets_csv="$(IFS=,; printf '%s' "${DATASET_LIST[*]}")"
  dataset_tag="$(IFS=_; printf '%s' "${DATASET_LIST[*]}")"

  log "Step 5/6: evaluate the shared gate"
  "$PYTHON_BIN" -m scripts.evaluation.run_shared_weight \
    --weights "$SHARED_WEIGHTS" \
    --datasets "$datasets_csv" \
    --encoder "$ENCODER" \
    --out "outputs/3expert_shared_weight_${dataset_tag}_${ENCODER}.json" \
    2>&1 | tee "logs/eval_shared_${dataset_tag}_${ENCODER}.log"

  if [ "$RUN_GATE_SELECTION" = "1" ]; then
    log "Step 6/6: select per-dataset gates on validation data"
    for dataset in "${DATASET_LIST[@]}"; do
      "$PYTHON_BIN" -m scripts.evaluation.run_gate_selection \
        --config "$(dataset_config "$dataset")" \
        --sasrec-model "outputs/models/${dataset}_sasrec.pt" \
        --itemcf-model "outputs/models/${dataset}_itemcf.json" \
        --semantic-emb "outputs/embeddings/${dataset}_${ENCODER}.npz" \
        --out-gate "outputs/gate_selection_${dataset}_${ENCODER}.json" \
        --out-test "outputs/3expert_${dataset}_${ENCODER}_val_selected.json" \
        2>&1 | tee "logs/gate_selection_${dataset}_${ENCODER}.log"
    done
  else
    log "Gate selection is optional; set RUN_GATE_SELECTION=1 to run it."
  fi

  if [ "$RUN_PAIRWISE_ABLATION" = "1" ]; then
    log "Running optional pairwise ablations after gate selection."
    for dataset in "${DATASET_LIST[@]}"; do
      require_file "outputs/gate_selection_${dataset}_${ENCODER}.json"
      "$PYTHON_BIN" -m scripts.evaluation.run_pairwise_ablation \
        --dataset "$dataset" \
        --encoder "$ENCODER" \
        --out "outputs/pairwise_ablation_${dataset}_${ENCODER}.json" \
        2>&1 | tee "logs/pairwise_ablation_${dataset}_${ENCODER}.log"
    done
  else
    log "Pairwise ablation is optional; set RUN_PAIRWISE_ABLATION=1 after gate selection."
  fi
}

log "LIME-Rec pipeline: datasets=${DATASETS}; device=${DEVICE}; seed=${SEED}"
step_download
step_build_itemcf
step_build_embeddings
step_train_sasrec
step_evaluate
log "Pipeline finished. Generated artifacts are under data/, outputs/, and logs/."
