#!/usr/bin/env bash
set -euo pipefail

DEVICE=${DEVICE:-cpu}
FROZEN_MODEL=${LLM_MODEL:-rule-based-mock}
RESULT_ROOT=${RESULT_ROOT:-output_final/results/agentic}
CACHE_ROOT=${CACHE_ROOT:-outputs/agentic/candidate_cache}
RECOVERY_OUT=${RECOVERY_OUT:-outputs/agentic/beauty_recovery_validation.pt}
PILOT_USERS=${PILOT_USERS:-1000}
COMMON=(--config configs/amazon_beauty.json
  --sasrec-model output_final/models/amazon_beauty_sasrec_seed0.pt
  --itemcf-model output_final/models/amazon_beauty_itemcf.json
  --semantic-emb output_final/models/amazon_beauty_bge_base.npz)

python -m scripts.agentic.build_candidate_cache "${COMMON[@]}" \
  --split validation --candidate-per-expert 20 --device "$DEVICE" \
  --out "$CACHE_ROOT/amazon_beauty_validation.jsonl"

python -m scripts.agentic.tune_agent_on_validation \
  --config configs/amazon_beauty.json \
  --candidate-cache "$CACHE_ROOT/amazon_beauty_validation.jsonl" \
  --recovery-out "$RECOVERY_OUT" \
  --frozen-config-out "$RESULT_ROOT/frozen_test_config.json" --model "$FROZEN_MODEL"

python -m scripts.agentic.build_candidate_cache "${COMMON[@]}" \
  --split test --candidate-per-expert 20 --device "$DEVICE" \
  --max-users "$PILOT_USERS" --sample-seed 2027 \
  --users-out "$RESULT_ROOT/beauty_test_users.txt" \
  --out "$CACHE_ROOT/amazon_beauty_test_${PILOT_USERS}.jsonl"
mkdir -p "$RESULT_ROOT/beauty"
cp "$RESULT_ROOT/beauty_test_users.txt" "$RESULT_ROOT/beauty/users.txt"
cp "$CACHE_ROOT/amazon_beauty_test_${PILOT_USERS}.jsonl.metadata.json" \
  "$RESULT_ROOT/beauty/candidate_metadata.json"

python -m scripts.agentic.run_agentic_pilot "${COMMON[@]}" \
  --candidate-cache "$CACHE_ROOT/amazon_beauty_test_${PILOT_USERS}.jsonl" \
  --frozen-config "$RESULT_ROOT/frozen_test_config.json" \
  --out-dir "$RESULT_ROOT/beauty" --device "$DEVICE" --resume

python -m scripts.agentic.bootstrap_agentic --dir "$RESULT_ROOT/beauty" \
  --n-resamples 1000 --seed 2027 --out "$RESULT_ROOT/paired_bootstrap.json"
python -m scripts.agentic.summarize_agentic --dir "$RESULT_ROOT/beauty" \
  --out-json "$RESULT_ROOT/agentic_summary.json" \
  --out-csv "$RESULT_ROOT/agentic_summary.csv"
