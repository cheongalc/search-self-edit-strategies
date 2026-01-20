#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  echo "Error: could not determine REPO_ROOT" >&2
  exit 1
fi
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/squad"

VLLM_HOST="$(hostname -i)"
VLLM_PORT=8001
VLLM_API_URL="http://${VLLM_HOST}:${VLLM_PORT}"

MODEL=$REPO_ROOT/squad/models/Qwen3-8B/baseline/rewrite/iter0 # model to use for data generation. For evaluation, set to the model to be evaluated. For RL training, set to the (n-1)'th RL checkpoint.
INSTRUCT_MODEL=true  # Set to true to enable instruct model flag
THINKING_MODE=true  # Set to true to enable thinking mode in chat templates

DATASET_IN="$REPO_ROOT/squad/data/squad_val.json"
DATASET_OUT="$REPO_ROOT/squad/data/Qwen3-8B/baseline/rewrite/iter1/val_thinking.json"

NUM_ARTICLES=200
START_INDEX=0
NUM_SELF_EDIT_TEMPLATES=1
NUM_COMPLETIONS_PER_TEMPLATE=1
MAX_NEW_TOKENS=12000
EXPLOIT_FRACTION=1.0

VLLM_LOG_PATH="$REPO_ROOT/squad/logs/Qwen3-8B/vllm_create_self_edits_baseline_rewrite_iter1_val.log"
mkdir -p "$(dirname "$VLLM_LOG_PATH")"

echo "Starting VLLM at URL ${VLLM_API_URL}"
echo "Logging to: $VLLM_LOG_PATH"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --max-model-len 16384 \
    --trust-remote-code \
    --generation-config vllm \
    --reasoning-parser qwen3 \
    --tensor-parallel-size 4 \
    >"$VLLM_LOG_PATH" 2>&1 &
VLLM_PID=$!

echo "vLLM started with PID $VLLM_PID"

# Wait for health-check
until curl --silent --fail ${VLLM_API_URL}/health >/dev/null; do sleep 3; done
echo "vLLM ready."

# Build instruct_model flag if enabled
INSTRUCT_MODEL_FLAG=""
if [ "$INSTRUCT_MODEL" = true ]; then
    INSTRUCT_MODEL_FLAG="--instruct_model"
fi

# Build thinking_mode flag if enabled
THINKING_MODE_FLAG=""
if [ "$THINKING_MODE" = true ]; then
    THINKING_MODE_FLAG="--thinking_mode"
fi

python3 -m "src.create_self_edits" \
    --vllm_api_url "$VLLM_API_URL" \
    --model "$MODEL" \
    $INSTRUCT_MODEL_FLAG \
    $THINKING_MODE_FLAG \
    --dataset_in "$DATASET_IN" \
    --dataset_out "$DATASET_OUT" \
    --num_articles "$NUM_ARTICLES" \
    --start_index "$START_INDEX" \
    --num_self_edit_templates "$NUM_SELF_EDIT_TEMPLATES" \
    --num_completions_per_template "$NUM_COMPLETIONS_PER_TEMPLATE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --baseline "rewrite"

echo "Shutting down vLLM"
kill $VLLM_PID

echo "Job finished."
