#!/usr/bin/env bash
# Prior-sensitivity witnesses under the matched-20 controlled protocol:
#   A) minilm semantic prior (all-MiniLM-L6-v2), full witness with calibration
#   B) no-semantic witness (experts seq,cf), with calibration
# 18 jobs on GPUs 0,1 (GRAM smoke runs on 2,3).
set -uo pipefail
cd "$(dirname "$0")/.."

PRED_DIR=output_final/results/controlled_gram/predictions
mkdir -p "$PRED_DIR" logs/priorsens

run_job() {
  local ds=$1 seed=$2 gpu=$3 variant=$4
  local extra=() label sem
  if [[ "$variant" == "minilm" ]]; then
    sem="output_final/models/amazon_${ds}_minilm.npz"
    label="LIME-Rec-matched20-minilm"
  else
    sem="output_final/models/amazon_${ds}_bge_base.npz"
    label="LIME-Rec-matched20-nosem"
    extra=(--experts seq,cf)
  fi
  CUDA_VISIBLE_DEVICES=$gpu python -m scripts.evaluation.run_controlled_lime_matched20 \
    --config "configs/amazon_${ds}.json" \
    --sasrec-model "outputs/models/controlled_gram/amazon_${ds}_sasrec_max20_seed${seed}.pt" \
    --itemcf-model "output_final/models/amazon_${ds}_itemcf.json" \
    --semantic-emb "$sem" \
    --seed "$seed" --device cuda "${extra[@]}" \
    --model-label "$label" \
    --predictions "$PRED_DIR/lime_${variant}_amazon_${ds}_seed${seed}.jsonl" \
    --out "output_final/results/controlled_gram/lime_${variant}_matched20/amazon_${ds}/seed${seed}.json" \
    > "logs/priorsens/${variant}_${ds}_seed${seed}.log" 2>&1
  echo "[done] ${variant} ${ds} seed${seed} gpu${gpu} rc=$?"
}

# wave over GPU 0 and GPU 1
(
  for ds in beauty toys sports; do for seed in 0 1 2; do run_job "$ds" "$seed" 0 minilm; done; done
) &
(
  for ds in beauty toys sports; do for seed in 0 1 2; do run_job "$ds" "$seed" 1 nosem; done; done
) &
wait
echo "[all-done] prior-sensitivity matched-20 runs"
