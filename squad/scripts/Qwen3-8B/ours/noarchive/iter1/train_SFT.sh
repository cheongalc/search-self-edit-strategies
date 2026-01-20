#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  echo "Error: could not determine REPO_ROOT" >&2
  exit 1
fi
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/squad"

SFT_DATASET_PATH=$REPO_ROOT/squad/data/Qwen3-8B/ours/iter2/sft_dataset_2025-11-29T22:42:27.176417_top2.jsonl
MODEL_NAME=$REPO_ROOT/squad/models/Qwen3-8B/ours/iter0
CHECKPOINT_OUTPUT_DIR=$REPO_ROOT/squad/models/Qwen3-8B/ours/iter1

PER_DEVICE_BATCH_SIZE=1
GRAD_ACC=5
EPOCHS=2
LR=3e-4
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.0
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LOGGING_STEPS=1

TRAINING_MAX_SEQ_LENGTH=3600

NUM_PROCESSES=2
DEEPSPEED_CONFIG_FILE=$REPO_ROOT/squad/config/deepspeed_stage3.json

# export NCCL_P2P_DISABLE=1  # fixes hangs on some setups

echo "Launching SFT run on $(hostname)..."
accelerate launch \
    --num_processes $NUM_PROCESSES \
    --deepspeed_config_file "$DEEPSPEED_CONFIG_FILE" \
    --mixed_precision bf16 \
    src/train_SFT.py \
    --sft_dataset_path "${SFT_DATASET_PATH}" \
    --model_name "${MODEL_NAME}" \
    --checkpoint_output_dir "${CHECKPOINT_OUTPUT_DIR}" \
    --per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    --logging_steps ${LOGGING_STEPS} \
    --training_max_seq_length ${TRAINING_MAX_SEQ_LENGTH}

echo "Job finished."
