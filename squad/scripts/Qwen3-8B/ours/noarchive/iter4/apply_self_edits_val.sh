#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  echo "Error: could not determine REPO_ROOT" >&2
  exit 1
fi
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/squad"

SELF_EDIT_DATA_PATH=$REPO_ROOT/squad/data/Qwen3-8B/ours/iter4/val_thinking.json
SCRATCH_DIR=/scratch
OUTPUT_DIR=$REPO_ROOT/squad/results
MODEL_NAME=$REPO_ROOT/squad/models/Qwen3-8B/ours/iter3
NUM_GPUS=4
TRAINING_MAX_SEQ_LENGTH=2048

PROGRESS_PORT=8900
VLLM_HOST="$(hostname -i)"
VLLM_API_PORT_START=8800

CHAIN_OF_THOUGHT=false

EVAL_TEMPERATURE=0.7
EVAL_TOP_P=0.9
EVAL_TOP_K=40
EVAL_MIN_P=0.0
EVAL_PRESENCE_PENALTY=1.5
EVAL_MAX_SEQ_LENGTH=2048
EVAL_TIMES=3

GRADER="anthropic"
ENV_FILE=$REPO_ROOT/.env

LOG_DIR=$REPO_ROOT/squad/logs/Qwen3-8B
LOG_LEVEL="DEBUG"
EXP_NAME="apply_self_edits_Qwen3-8B_iter4_val"

EXECUTORS_PER_GPU=1
EVAL_WORKERS_PER_GPU=2

# Build chain_of_thought flag if enabled
CHAIN_OF_THOUGHT_FLAG=""
if [ "$CHAIN_OF_THOUGHT" = true ]; then
    CHAIN_OF_THOUGHT_FLAG="--chain_of_thought"
fi

python3 -m "src.apply_self_edits" \
    --self_edit_data_path "$SELF_EDIT_DATA_PATH" \
    --scratch_dir "$SCRATCH_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_gpus "$NUM_GPUS" \
    --training_max_seq_length "$TRAINING_MAX_SEQ_LENGTH" \
    --progress_port "$PROGRESS_PORT" \
    --vllm_host "$VLLM_HOST" \
    --vllm_api_port_start "$VLLM_API_PORT_START" \
    $CHAIN_OF_THOUGHT_FLAG \
    --eval_temperature "$EVAL_TEMPERATURE" \
    --eval_top_p "$EVAL_TOP_P" \
    --eval_top_k "$EVAL_TOP_K" \
    --eval_min_p "$EVAL_MIN_P" \
    --eval_presence_penalty "$EVAL_PRESENCE_PENALTY" \
    --eval_max_seq_length "$EVAL_MAX_SEQ_LENGTH" \
    --eval_times "$EVAL_TIMES" \
    --executors_per_gpu "$EXECUTORS_PER_GPU" \
    --eval_workers_per_gpu "$EVAL_WORKERS_PER_GPU" \
    --grader "$GRADER" \
    --env_file "$ENV_FILE" \
    --log_dir "$LOG_DIR" \
    --log_level "$LOG_LEVEL" \
    --exp_name "$EXP_NAME" \
    --resume

echo "Job finished."
