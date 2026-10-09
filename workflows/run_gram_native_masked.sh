#!/usr/bin/env bash
# Native repeat-masked GRAM inference (test-only) for all 9 controlled runs.
# Loads the exact epoch-30 checkpoints that produced the repeat-allowed
# predictions; only the decoding constraint changes (history items excluded).
set -uo pipefail
cd "$(dirname "$0")/.."
GRAMPY="$(pwd)/.conda/gram/bin/python"

declare -A RUN_DIR=(
  ["beauty_0"]="Beauty/0_20260916_1206" ["beauty_1"]="Beauty/1_20260918_0506" ["beauty_2"]="Beauty/2_20260918_2149"
  ["sports_0"]="Sports/0_20260917_0147" ["sports_1"]="Sports/1_20260917_0453" ["sports_2"]="Sports/2_20260918_0200"
  ["toys_0"]="Toys/0_20260916_1217" ["toys_1"]="Toys/1_20260919_0214" ["toys_2"]="Toys/2_20260919_1432"
)

RESULT_ROOT=output_final/results/controlled_gram
mkdir -p logs/gram_mask "$RESULT_ROOT/predictions"

run_one() {
  local ds=$1 seed=$2 gpus=$3 port=$4
  local gram_name; gram_name=$(python -c "print('${ds}'.title())")
  local clusters id_len num_cf
  case "$ds" in
    beauty) clusters=128; id_len=7; num_cf=10 ;;
    toys)   clusters=32;  id_len=5; num_cf=5  ;;
    sports) clusters=32;  id_len=7; num_cf=10 ;;
  esac
  local ckpt="../log/${RUN_DIR[${ds}_${seed}]}/id_0_rec_30/model_rec_phase_1_epoch_30.pt"
  local marker; marker=$(mktemp /tmp/gram-mask-marker.XXXXXX)
  (cd external/GRAM/command && CUDA_VISIBLE_DEVICES=$gpus "$GRAMPY" \
    ../src/main_generative_gram.py --datasets "$gram_name" --distributed 1 \
    --master_port "$port" --gpu 0,1 --seed "$seed" --train 0 \
    --rec_model_path "$ckpt" \
    --item_prompt_max_len 128 --item_prompt all_text --cf_model sasrec --id_linking 1 \
    --max_his 20 --rec_batch_size 32 --gradient_accumulation_steps 2 --rec_lr 1e-3 \
    --rec_epochs 30 --test_epoch_rec 5 --save_rec_epochs 5 --save_predictions 1 \
    --top_k_similar_item "$num_cf" --item_id_type split \
    --hierarchical_id_type "hierarchy_v1_c${clusters}_l${id_len}_len32768_split" \
    --mask_history_items 1) > "logs/gram_mask/${ds}_seed${seed}.log" 2>&1
  local rc=$?
  local prediction
  prediction=$(find external/GRAM/preds -type f -newer "$marker" -name "*${gram_name}*pred_test_all.tsv" | sort | tail -1)
  rm -f "$marker"
  if [[ -z "$prediction" ]]; then
    echo "[fail] $ds seed$seed rc=$rc NO_PREDICTION"; return 1
  fi
  python -m scripts.baselines.import_gram_predictions \
    --config "configs/amazon_${ds}.json" --gram-tsv "$prediction" \
    --gram-user-sequence "external/GRAM/rec_datasets/$gram_name/user_sequence.txt" \
    --semantic-mapping "external/GRAM/rec_datasets/$gram_name/item_generative_indexing_hierarchy_v1_c${clusters}_l${id_len}_len32768_split.txt" \
    --seed "$seed" \
    --out "$RESULT_ROOT/predictions/gram_nativemask_amazon_${ds}_seed${seed}.jsonl" \
    --audit-out "$RESULT_ROOT/gram_nativemask/amazon_${ds}/seed${seed}_mapping.json" >> "logs/gram_mask/${ds}_seed${seed}.log" 2>&1
  echo "[done] mask $ds seed$seed rc=$rc pred=$(basename "$prediction")"
}

if [[ -n "${GRAM_MASK_JOBS_FILE:-}" ]]; then
  mapfile -t job_list < "$GRAM_MASK_JOBS_FILE"
else
  job_list=(
    "beauty 0" "beauty 1" "beauty 2"
    "toys 0" "toys 1" "toys 2"
    "sports 0" "sports 1" "sports 2"
  )
fi
i=0
for job in "${job_list[@]}"; do
  read -r ds seed <<< "$job"
  if (( i % 2 == 0 )); then gpus="0,1"; port=$((31000 + i)); else gpus="2,3"; port=$((32000 + i)); fi
  run_one "$ds" "$seed" "$gpus" "$port" &
  if (( i % 2 == 1 )); then wait; fi
  i=$((i + 1))
done
wait
echo "[all-done] native repeat-masked GRAM sweep"
