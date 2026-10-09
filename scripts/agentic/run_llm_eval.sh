#!/usr/bin/env bash
# vLLM 服务一个模型（Qwen3-4B 或 selector_merged）+ 对指定 (domain, seed, kind) 跑 evaluate --llm
# usage:
#   serve:  bash run_llm_eval.sh serve <model_dir> <served_name> <gpu_id> <port> <logfile>
#   eval:   bash run_llm_eval.sh eval <domain> <seed> <service_port> <served_name> <artifact_json> <out_dir>
set -euo pipefail
cd "$(dirname "$0")/../.."

CMD=$1

if [ "$CMD" = serve ]; then
  MODEL_DIR=$2; SERVED_NAME=$3; GPU=$4; PORT=$5; LOGF=$6
  CUDA_VISIBLE_DEVICES=$GPU nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-model-len 32768 --port "$PORT" --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 > "$LOGF" 2>&1 &
  echo "serve pid $! ($SERVED_NAME on gpu $GPU port $PORT)"

elif [ "$CMD" = eval ]; then
  DOMAIN=$2; TRAIN_SEED=$3; PORT=$4; SERVED_NAME=$5; ARTIFACT=$6; OUT_DIR=$7
  EXP="outputs/budgeted_tools_v1/amazon_${DOMAIN}/seed${TRAIN_SEED}"
  test ! -e "$OUT_DIR" || { echo "eval output exists: $OUT_DIR"; exit 1; }
  export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
  export LLM_API_KEY=local
  export LLM_MODEL="$SERVED_NAME"
  python -m scripts.agentic.run_budgeted_tools evaluate \
    --data "$EXP/test.json" --router "$EXP/fitted/router.json" \
    --llm --max-tokens 128 --bootstrap-resamples 10000 --seed 2027 \
    --llm-artifact "$ARTIFACT" \
    --out-dir "$OUT_DIR"

else
  echo "unknown command"; exit 1
fi