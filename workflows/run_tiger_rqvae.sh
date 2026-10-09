#!/usr/bin/env bash
# TIGER (second generative target) — Stage 1: vanilla RQ-VAE tokenizers on
# frozen BGE item embeddings (alpha=0, beta=0: pure semantic RQ-VAE, no
# collaborative/diversity regularization), one per dataset.
set -uo pipefail
cd "$(dirname "$0")/.."
GRAMPY="$(pwd)/.conda/gram/bin/python"
mkdir -p external/LETTER/checkpoint_lime logs/tiger

run_rqvae() {
  local ds=$1 gpu=$2
  local DS; DS=$(python -c "print('${ds}'.title())")
  (cd external/LETTER/RQ-VAE && CUDA_VISIBLE_DEVICES=$gpu "$GRAMPY" main.py \
    --data_path "../data_lime/${DS}/${DS}.emb-bge.npy" \
    --alpha 0.0 --beta 0.0 \
    --epochs 2000 --eval_step 250 --batch_size 8192 \
    --ckpt_dir "../checkpoint_lime/${DS}" \
    --device cuda:0) > "logs/tiger/rqvae_${ds}.log" 2>&1
  echo "[done] rqvae $ds rc=$?"
}

run_rqvae beauty 0 &
run_rqvae toys 1 &
run_rqvae sports 2 &
wait
echo "[all-done] RQ-VAE tokenizers"
