#!/usr/bin/env bash
# No-calibration recovery witness runs (matched-20 protocol, penalty forced to 0).
# 9 jobs = {beauty,toys,sports} x {0,1,2}, distributed over 4 GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."

PRED_DIR=output_final/results/controlled_gram/predictions
OUT_DIR=output_final/results/controlled_gram/lime_nocal_matched20
mkdir -p "$PRED_DIR" "$OUT_DIR" logs/nocal

run_job() {
  local ds=$1 seed=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu python -m scripts.evaluation.run_controlled_lime_matched20 \
    --config "configs/amazon_${ds}.json" \
    --sasrec-model "outputs/models/controlled_gram/amazon_${ds}_sasrec_max20_seed${seed}.pt" \
    --itemcf-model "output_final/models/amazon_${ds}_itemcf.json" \
    --semantic-emb "output_final/models/amazon_${ds}_bge_base.npz" \
    --seed "$seed" --device cuda --disable-calibration \
    --model-label "LIME-Rec-matched20-nocal" \
    --predictions "$PRED_DIR/lime_nocal_amazon_${ds}_seed${seed}.jsonl" \
    --out "$OUT_DIR/amazon_${ds}/seed${seed}.json" \
    > "logs/nocal/${ds}_seed${seed}.log" 2>&1
  echo "[done] ${ds} seed${seed} gpu${gpu} rc=$?"
}

# GPU 0: beauty 0,1,2 ; GPU 1: toys 0,1,2 ; GPU 2: sports 0,1 ; GPU 3: sports 2
(
  run_job beauty 0 0
  run_job beauty 1 0
  run_job beauty 2 0
) &
(
  run_job toys 0 1
  run_job toys 1 1
  run_job toys 2 1
) &
(
  run_job sports 0 2
  run_job sports 1 2
) &
(
  run_job sports 2 3
) &
wait
echo "[all-done] no-cal matched-20 runs"
