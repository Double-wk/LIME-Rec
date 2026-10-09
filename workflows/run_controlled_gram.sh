#!/usr/bin/env bash
set -euo pipefail

DEVICE=${DEVICE:-cuda}
GRAM_ROOT=${GRAM_ROOT:-external/GRAM}
RESULT_ROOT=output_final/results/controlled_gram
MODEL_ROOT=outputs/models/controlled_gram
mkdir -p "$MODEL_ROOT" "$RESULT_ROOT/alignment" "$RESULT_ROOT/predictions"

python -m scripts.baselines.prepare_gram --gram-root "$GRAM_ROOT" \
  --out "$RESULT_ROOT/provenance.json"

for dataset in beauty toys sports; do
  gram_name=$(python -c "print('${dataset}'.title())")
  case "$dataset" in
    beauty) alignment_mapping=item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt ;;
    toys) alignment_mapping=item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt ;;
    sports) alignment_mapping=item_generative_indexing_hierarchy_v1_c32_l7_len32768_split.txt ;;
  esac
  python -m scripts.baselines.audit_gram_alignment \
    --config "configs/amazon_${dataset}.json" --gram-root "$GRAM_ROOT" \
    --gram-user-sequence "$GRAM_ROOT/rec_datasets/$gram_name/user_sequence.txt" \
    --semantic-mapping "$GRAM_ROOT/rec_datasets/$gram_name/$alignment_mapping" \
    --out "$RESULT_ROOT/alignment/amazon_${dataset}.json"
  for seed in 0 1 2; do
    checkpoint="$MODEL_ROOT/amazon_${dataset}_sasrec_max20_seed${seed}.pt"
    if [[ ! -f "$checkpoint" ]]; then
      python -m scripts.training.train_sasrec --config "configs/amazon_${dataset}.json" \
        --output "$checkpoint" --maxlen 20 --hidden 64 --layers 2 --heads 2 \
        --dropout 0.2 --loss ce --validation-no-mask --seed "$seed" --device "$DEVICE"
    fi
    python -m scripts.evaluation.run_controlled_lime_matched20 \
      --config "configs/amazon_${dataset}.json" --sasrec-model "$checkpoint" \
      --itemcf-model "output_final/models/amazon_${dataset}_itemcf.json" \
      --semantic-emb "output_final/models/amazon_${dataset}_bge_base.npz" \
      --seed "$seed" --device "$DEVICE" \
      --predictions "$RESULT_ROOT/predictions/lime_amazon_${dataset}_seed${seed}.jsonl" \
      --out "$RESULT_ROOT/lime_matched20/amazon_${dataset}/seed${seed}.json"
  done
done

if [[ "${RUN_GRAM:-1}" == "0" ]]; then
  echo "[controlled] LIME matched-20 runs complete; GRAM skipped by RUN_GRAM=0."
  exit 0
fi

if [[ -z "${GRAM_PYTHON:-}" ]]; then
  echo "Set GRAM_PYTHON to the Python executable in the isolated GRAM environment."
  exit 5
fi

# Fail-safe default: only Beauty seed 0. Expand after its alignment/import/evaluation passes:
# GRAM_DATASETS="beauty toys sports" GRAM_SEEDS="0 1 2" workflows/run_controlled_gram.sh
for dataset in ${GRAM_DATASETS:-beauty}; do
  case "$dataset" in
    beauty) gram_name=Beauty; clusters=128; id_len=7; num_cf=10 ;;
    toys) gram_name=Toys; clusters=32; id_len=5; num_cf=5 ;;
    sports) gram_name=Sports; clusters=32; id_len=7; num_cf=10 ;;
    *) echo "unknown GRAM dataset: $dataset"; exit 6 ;;
  esac
  mapping="item_generative_indexing_hierarchy_v1_c${clusters}_l${id_len}_len32768_split.txt"
  for seed in ${GRAM_SEEDS:-0}; do
    marker=$(mktemp /tmp/gram-run-marker.XXXXXX)
    (cd "$GRAM_ROOT/command" && CUDA_VISIBLE_DEVICES=${GRAM_GPUS:-0,1} "$GRAM_PYTHON" \
      ../src/main_generative_gram.py --datasets "$gram_name" --distributed 1 \
      --master_port "$((2341 + seed))" --gpu "${GRAM_GPUS:-0,1}" --seed "$seed" --train 1 \
      --item_prompt_max_len 128 --item_prompt all_text --cf_model sasrec --id_linking 1 \
      --max_his 20 --rec_batch_size 32 --gradient_accumulation_steps 2 --rec_lr 1e-3 \
      --rec_epochs 30 --test_epoch_rec 5 --save_rec_epochs 5 --save_predictions 1 \
      --top_k_similar_item "$num_cf" --item_id_type split \
      --hierarchical_id_type "hierarchy_v1_c${clusters}_l${id_len}_len32768_split")
    prediction=$(find "$GRAM_ROOT/preds" -type f -newer "$marker" -name "*${gram_name}*pred_test_all.tsv" | sort | tail -1)
    rm -f "$marker"
    if [[ -z "$prediction" ]]; then echo "BLOCKED_BY_MISSING_GRAM_PREDICTION"; exit 7; fi
    imported="$RESULT_ROOT/predictions/gram_amazon_${dataset}_seed${seed}.jsonl"
    python -m scripts.baselines.import_gram_predictions \
      --config "configs/amazon_${dataset}.json" --gram-tsv "$prediction" \
      --gram-user-sequence "$GRAM_ROOT/rec_datasets/$gram_name/user_sequence.txt" \
      --semantic-mapping "$GRAM_ROOT/rec_datasets/$gram_name/$mapping" --seed "$seed" \
      --out "$imported" --audit-out "$RESULT_ROOT/gram/amazon_${dataset}/seed${seed}_mapping.json"
    python -m scripts.baselines.evaluate_controlled --config "configs/amazon_${dataset}.json" \
      --predictions "$imported" --alignment "$RESULT_ROOT/alignment/amazon_${dataset}.json" \
      --out "$RESULT_ROOT/gram/amazon_${dataset}/seed${seed}.json"
  done
done

if [[ "${GRAM_DATASETS:-beauty}" == "beauty toys sports" && "${GRAM_SEEDS:-0}" == "0 1 2" ]]; then
  python -m scripts.baselines.summarize_controlled --root "$RESULT_ROOT"
fi
