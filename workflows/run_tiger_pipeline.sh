#!/usr/bin/env bash
# TIGER (LETTER implementation) full pipeline after RQ-VAE tokenizers:
#   stage 1: generate_indices per dataset (collision-resolved semantic IDs)
#   stage 2: T5 finetune, 3 datasets x seeds 0/1/2
#   stage 3: export per-user top-20 rankings in audit format
set -uo pipefail
cd "$(dirname "$0")/.."
GRAMPY="$(pwd)/.conda/gram/bin/python"
RQVAE_CKPT_ROOT=external/LETTER/checkpoint_lime

stage_indices() {
  local ds=$1 gpu=$2
  local DS; DS=$(python -c "print('${ds}'.title())")
  local ckpt
  ckpt=$(ls -t "$RQVAE_CKPT_ROOT/$DS"/*/best_collision_model.pth 2>/dev/null | head -1)
  if [[ -z "$ckpt" ]]; then echo "[fail] no RQ-VAE ckpt for $DS"; return 1; fi
  mkdir -p external/LETTER/RQ-VAE/checkpoint/alpha0.0-beta0.0
  cp "$ckpt" "external/LETTER/RQ-VAE/checkpoint/alpha0.0-beta0.0/${DS}_model.pth"
  (cd external/LETTER/RQ-VAE && ln -sfn ../data_lime data && CUDA_VISIBLE_DEVICES=$gpu "$GRAMPY" \
    generate_indices.py --dataset "$DS" --root_path ./checkpoint/ \
    --alpha 0.0 --beta 0.0 --epoch 2000 --checkpoint "${DS}_model.pth") \
    > "logs/tiger/indices_${ds}.log" 2>&1
  # move generated index to the dataset dir as the canonical .index.json
  local gen="external/LETTER/data_lime/${DS}/${DS}.index.epoch2000.alpha0.0-beta0.0.json"
  if [[ ! -f "$gen" ]]; then echo "[fail] index not generated for $DS"; return 1; fi
  cp "$gen" "external/LETTER/data_lime/${DS}/${DS}.index.json"
  echo "[done] indices $DS"
}

stage_finetune() {
  local ds=$1 seed=$2 gpu=$3
  local DS; DS=$(python -c "print('${ds}'.title())")
  (cd external/LETTER/LETTER-TIGER && CUDA_VISIBLE_DEVICES=$gpu "$GRAMPY" finetune.py \
    --output_dir "ckpt_lime/${DS}_seed${seed}" \
    --dataset "$DS" --data_path ../data_lime \
    --per_device_batch_size 256 --learning_rate 5e-4 --epochs 200 \
    --index_file .index.json --temperature 1.0 --seed "$seed") \
    > "logs/tiger/finetune_${ds}_seed${seed}.log" 2>&1
  echo "[done] finetune $ds seed$seed rc=$?"
}

stage_export() {
  local ds=$1 seed=$2 gpu=$3
  local DS; DS=$(python -c "print('${ds}'.title())")
  CUDA_VISIBLE_DEVICES=$gpu python -m scripts.baselines.export_tiger_predictions \
    --dataset "$DS" --ckpt-path "external/LETTER/LETTER-TIGER/ckpt_lime/${DS}_seed${seed}" \
    --mapping "external/LETTER/data_lime/${DS}/mapping.json" --seed "$seed" \
    --out "output_final/results/controlled_gram/predictions/tiger_amazon_${ds}_seed${seed}.jsonl" \
    > "logs/tiger/export_${ds}_seed${seed}.log" 2>&1
  echo "[done] export $ds seed$seed rc=$?"
}

case "${1:-all}" in
  indices)
    stage_indices beauty 0 & stage_indices toys 1 & stage_indices sports 2 & wait ;;
  finetune)
    i=0
    for ds in beauty toys sports; do
      for seed in 0 1 2; do
        gpu=$((i % 4))
        stage_finetune "$ds" "$seed" "$gpu" &
        if (( i % 4 == 3 )); then wait; fi
        i=$((i + 1))
      done
    done
    wait ;;
  export)
    i=0
    for ds in beauty toys sports; do
      for seed in 0 1 2; do
        gpu=$((i % 4))
        stage_export "$ds" "$seed" "$gpu" &
        if (( i % 4 == 3 )); then wait; fi
        i=$((i + 1))
      done
    done
    wait ;;
  *) echo "usage: $0 [indices|finetune|export]"; exit 2 ;;
esac
echo "[all-done] stage $1"
