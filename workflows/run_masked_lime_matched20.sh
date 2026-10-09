#!/usr/bin/env bash
# LIME-Rec repeat-masked witness under the matched-20 controlled protocol:
# same capped 20-item history window is excluded from candidates
# (mirrors the native repeat-masked GRAM decoding condition).
set -uo pipefail
cd "$(dirname "$0")/.."

PRED_DIR=output_final/results/controlled_gram/predictions
mkdir -p "$PRED_DIR" logs/masked_lime

run_job() {
  local ds=$1 seed=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu python -m scripts.evaluation.run_controlled_lime_matched20 \
    --config "configs/amazon_${ds}.json" \
    --sasrec-model "outputs/models/controlled_gram/amazon_${ds}_sasrec_max20_seed${seed}.pt" \
    --itemcf-model "output_final/models/amazon_${ds}_itemcf.json" \
    --semantic-emb "output_final/models/amazon_${ds}_bge_base.npz" \
    --seed "$seed" --device cuda --mask-history \
    --model-label "LIME-Rec-matched20-masked" \
    --predictions "$PRED_DIR/lime_masked_amazon_${ds}_seed${seed}.jsonl" \
    --out "output_final/results/controlled_gram/lime_masked_matched20/amazon_${ds}/seed${seed}.json" \
    > "logs/masked_lime/${ds}_seed${seed}.log" 2>&1
  echo "[done] masked-lime ${ds} seed${seed} gpu${gpu} rc=$?"
}

(
  run_job beauty 0 0; run_job beauty 1 0; run_job beauty 2 0
) &
(
  run_job toys 0 1; run_job toys 1 1; run_job toys 2 1
) &
(
  run_job sports 0 2; run_job sports 1 2
) &
(
  run_job sports 2 3
) &
wait
echo "[all-done] repeat-masked LIME matched-20 runs"
