#!/usr/bin/env bash
# Controlled GRAM-on-Yelp audit (added 2026-09-30), modeled line-by-line on
# workflows/run_controlled_gram.sh. GRAM's official Yelp configuration differs
# from the Amazon domains only in the semantic-ID hierarchy (c32_l9_len128,
# top_k_similar_item=5; see external/GRAM/command/train_gram_yelp.sh); splits,
# history window, batch, epochs, and evaluator are the frozen controlled
# protocol. data/yelp/user_sequence.txt is byte-identical to
# external/GRAM/rec_datasets/Yelp/user_sequence.txt (30,431 users).
#
# Stages:
#   STAGE=lime  alignment audit + matched-20 SASRec + LIME-Rec-20 witness (seeds 0-2)
#   STAGE=gram  GRAM training for ONE seed: GRAM_SEED=n GRAM_GPUS=0,1 GRAM_PORT=2341
#   STAGE=post  TSV import + shared evaluation for ONE seed: GRAM_SEED=n
#   STAGE=audit canonical paired recovery audit over all seeds
set -euo pipefail

DEVICE=${DEVICE:-cuda}
STAGE=${STAGE:-lime}
GRAM_ROOT=${GRAM_ROOT:-external/GRAM}
RESULT_ROOT=output_final/results/controlled_gram_yelp
MODEL_ROOT=outputs/models/controlled_gram_yelp
CONFIG=configs/yelp.json
GRAM_NAME=Yelp
CLUSTERS=32
ID_LEN=9
NUM_CF=5
MAPPING="item_generative_indexing_hierarchy_v1_c${CLUSTERS}_l${ID_LEN}_len128_split.txt"
SEMANTIC_EMB=${SEMANTIC_EMB:-outputs/embeddings/yelp_bge_base.npz}

mkdir -p "$MODEL_ROOT" "$RESULT_ROOT/alignment" "$RESULT_ROOT/predictions" \
  "$RESULT_ROOT/lime_matched20/yelp" "$RESULT_ROOT/gram/yelp"

if [[ "$STAGE" == "lime" ]]; then
  python -m scripts.baselines.audit_gram_alignment \
    --config "$CONFIG" --gram-root "$GRAM_ROOT" \
    --gram-user-sequence "$GRAM_ROOT/rec_datasets/$GRAM_NAME/user_sequence.txt" \
    --semantic-mapping "$GRAM_ROOT/rec_datasets/$GRAM_NAME/$MAPPING" \
    --out "$RESULT_ROOT/alignment/yelp.json"
  for seed in 0 1 2; do
    checkpoint="$MODEL_ROOT/yelp_sasrec_max20_seed${seed}.pt"
    if [[ ! -f "$checkpoint" ]]; then
      python -m scripts.training.train_sasrec --config "$CONFIG" \
        --output "$checkpoint" --maxlen 20 --hidden 64 --layers 2 --heads 2 \
        --dropout 0.2 --loss ce --validation-no-mask --seed "$seed" --device "$DEVICE"
    fi
    python -m scripts.evaluation.run_controlled_lime_matched20 \
      --config "$CONFIG" --sasrec-model "$checkpoint" \
      --itemcf-model "output_final/models/yelp_itemcf.json" \
      --semantic-emb "$SEMANTIC_EMB" \
      --seed "$seed" --device "$DEVICE" \
      --predictions "$RESULT_ROOT/predictions/lime_yelp_seed${seed}.jsonl" \
      --out "$RESULT_ROOT/lime_matched20/yelp/seed${seed}.json"
  done
  echo "[yelp] LIME matched-20 stage complete."
  exit 0
fi

if [[ "$STAGE" == "gram" || "$STAGE" == "post" ]]; then
  seed=${GRAM_SEED:?GRAM_SEED required}
  if [[ -z "${GRAM_PYTHON:-}" ]]; then
    echo "Set GRAM_PYTHON to the Python executable in the isolated GRAM environment."
    exit 5
  fi
  if [[ "$STAGE" == "gram" ]]; then
    marker=$(mktemp /tmp/gram-yelp-marker.XXXXXX)
    (cd "$GRAM_ROOT/command" && CUDA_VISIBLE_DEVICES=${GRAM_GPUS:-0,1} "$GRAM_PYTHON" \
      ../src/main_generative_gram.py --datasets "$GRAM_NAME" --distributed 1 \
      --master_port "${GRAM_PORT:-$((2341 + seed))}" --gpu "${GRAM_GPUS:-0,1}" \
      --seed "$seed" --train 1 \
      --item_prompt_max_len 128 --item_prompt all_text --cf_model sasrec --id_linking 1 \
      --max_his 20 --rec_batch_size 32 --gradient_accumulation_steps 2 --rec_lr 1e-3 \
      --rec_epochs 30 --test_epoch_rec 5 --save_rec_epochs 5 --save_predictions 1 \
      --top_k_similar_item "$NUM_CF" --item_id_type split \
      --hierarchical_id_type "hierarchy_v1_c${CLUSTERS}_l${ID_LEN}_len128_split")
    prediction=$(find "$GRAM_ROOT/preds" -type f -newer "$marker" -name "*${GRAM_NAME}*pred_test_all.tsv" | sort | tail -1)
    rm -f "$marker"
    if [[ -z "$prediction" ]]; then echo "BLOCKED_BY_MISSING_GRAM_PREDICTION"; exit 7; fi
    echo "[yelp] GRAM seed $seed trained; TSV: $prediction"
  fi
  prediction=$(find "$GRAM_ROOT/preds" -type f -name "*${GRAM_NAME}*pred_test_all.tsv" | sort | tail -1)
  python -m scripts.baselines.import_gram_predictions \
    --config "$CONFIG" --gram-tsv "$prediction" \
    --gram-user-sequence "$GRAM_ROOT/rec_datasets/$GRAM_NAME/user_sequence.txt" \
    --semantic-mapping "$GRAM_ROOT/rec_datasets/$GRAM_NAME/$MAPPING" --seed "$seed" \
    --out "$RESULT_ROOT/predictions/gram_yelp_seed${seed}.jsonl" \
    --audit-out "$RESULT_ROOT/gram/yelp/seed${seed}_mapping.json"
  python -m scripts.baselines.evaluate_controlled --config "$CONFIG" \
    --predictions "$RESULT_ROOT/predictions/gram_yelp_seed${seed}.jsonl" \
    --alignment "$RESULT_ROOT/alignment/yelp.json" \
    --out "$RESULT_ROOT/gram/yelp/seed${seed}.json"
  echo "[yelp] post stage complete for seed $seed."
  exit 0
fi

if [[ "$STAGE" == "audit" ]]; then
  python -m scripts.evaluation.recovery_audit_gram \
    --predictions-dir "$RESULT_ROOT/predictions" \
    --gram-prefix gram --lime-prefix lime \
    --datasets yelp \
    --comparison "LIME-Rec-20 minus retrained GRAM on Yelp (per-seed paired)" \
    --out-json "$RESULT_ROOT/recovery_audit_gram_yelp.json"
  echo "[yelp] audit stage complete."
  exit 0
fi

echo "unknown STAGE: $STAGE"; exit 8
