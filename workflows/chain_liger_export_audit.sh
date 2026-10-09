#!/usr/bin/env bash
# Chain: wait for LIGER seed-0 training runs to finish -> export test rankings
# (official beam=100) -> shared evaluator -> paired recovery audit vs the
# LIME-Rec-20 witness. Also launches the sports seed-0 training run as soon
# as a GPU frees.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled

RES=output_final/results/controlled_liger
mkdir -p "$RES/predictions" "$RES/liger" "$RES/audit"

wait_for() {  # wait_for <dataset-name-in-ps>
  while pgrep -f "run_liger_target --dataset $1 " > /dev/null; do sleep 120; done
}

for ds in beauty toys; do
  wait_for "$ds"
  gpu=6; [[ "$ds" == "toys" ]] && gpu=7
  echo "[chain] $(date +%H:%M) exporting $ds (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m scripts.baselines.export_liger_predictions \
    --dataset "$ds" --seed 0 --method setting --mode liger --device-id 0 \
    --out "$RES/predictions/liger_amazon_${ds}_seed0.jsonl" \
    --num-beams 100 --test-batch-size 32 \
    > "/tmp/liger_export_${ds}.log" 2>&1
  .venv/bin/python -m scripts.baselines.evaluate_controlled \
    --config "configs/amazon_${ds}.json" \
    --predictions "$RES/predictions/liger_amazon_${ds}_seed0.jsonl" \
    --alignment "output_final/results/controlled_gram/alignment/amazon_${ds}.json" \
    --out "$RES/liger/amazon_${ds}/seed0.json" \
    > "/tmp/liger_eval_${ds}.log" 2>&1
  echo "[chain] $(date +%H:%M) $ds evaluated: $(cat "$RES/liger/amazon_${ds}/seed0.json" 2>/dev/null | head -c 200)"
done

# paired audit vs the canonical witness predictions
for ds in beauty toys; do
  cp "output_final/results/controlled_gram/predictions/lime_amazon_${ds}_seed0.jsonl" \
     "$RES/predictions/" 2>/dev/null
done
.venv/bin/python -m scripts.evaluation.recovery_audit_gram \
  --predictions-dir "$RES/predictions" \
  --gram-prefix liger --lime-prefix lime \
  --datasets amazon_beauty,amazon_toys \
  --comparison "LIME-Rec-20 minus LIGER (seed 0, per-seed paired)" \
  --out-json "$RES/audit/recovery_audit_liger_seed0.json" \
  > /tmp/liger_audit.log 2>&1
echo "[chain] $(date +%H:%M) audit done: $(cat /tmp/liger_audit.log | tail -1)"

# launch sports seed-0 training on the freed GPU
if [[ ! -f external/LIGER/results/liger/Amazon_Sports_and_Outdoors/lime_setting_Sports_and_Outdoors_seed0_seed_0/results/ckpt_best.pt ]]; then
  echo "[chain] $(date +%H:%M) launching sports seed0 training on gpu 7"
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m scripts.baselines.run_liger_target \
    --dataset sports --seed 0 --method setting --mode liger --device-id 0 \
    > /tmp/liger_sports_s0.log 2>&1
fi
echo "[chain] $(date +%H:%M) chain complete"
