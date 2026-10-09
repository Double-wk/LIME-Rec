#!/usr/bin/env bash
# Frozen-weight repeat-masked LIME witness: gate fitted under the repeat-allowed
# protocol, evaluated with history-masked scores (sensitivity diagnostic paired
# with native repeat-masked GRAM decoding).
set -uo pipefail
cd "$(dirname "$0")/.."

PRED_DIR=output_final/results/controlled_gram/predictions
mkdir -p "$PRED_DIR" logs/masked_lime_frozen

run_job() {
  local ds=$1 seed=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu python -m scripts.evaluation.run_controlled_lime_matched20 \
    --config "configs/amazon_${ds}.json" \
    --sasrec-model "outputs/models/controlled_gram/amazon_${ds}_sasrec_max20_seed${seed}.pt" \
    --itemcf-model "output_final/models/amazon_${ds}_itemcf.json" \
    --semantic-emb "output_final/models/amazon_${ds}_bge_base.npz" \
    --seed "$seed" --device cuda --eval-mask-history \
    --model-label "LIME-Rec-matched20-frozengate-evalmasked" \
    --predictions "$PRED_DIR/lime_maskfrozen_amazon_${ds}_seed${seed}.jsonl" \
    --out "output_final/results/controlled_gram/lime_maskfrozen_matched20/amazon_${ds}/seed${seed}.json" \
    > "logs/masked_lime_frozen/${ds}_seed${seed}.log" 2>&1
  echo "[done] maskfrozen ${ds} seed${seed} gpu${gpu} rc=$?"
}

(
  run_job beauty 0 2; run_job beauty 1 2; run_job beauty 2 2
  run_job toys 0 2; run_job toys 1 2; run_job toys 2 2
) &
(
  run_job sports 0 3; run_job sports 1 3; run_job sports 2 3
) &
wait
echo "[all-done] frozen-gate repeat-masked LIME runs"
