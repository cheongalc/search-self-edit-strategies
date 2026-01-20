#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  echo "Error: could not determine REPO_ROOT" >&2
  exit 1
fi
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/squad"

SELF_EDIT_RESULTS_JSON_PATH=$REPO_ROOT/squad/results/Qwen3-8B/baseline/rewrite/iter2_train/results.json
DATASET_OUTPUT_DIR=$REPO_ROOT/squad/data/Qwen3-8B/baseline/rewrite/iter3
K_BEST=2

python3 -m "src.build_SFT_dataset" \
    --self_edit_results_json_path "$SELF_EDIT_RESULTS_JSON_PATH" \
    --dataset_output_dir "$DATASET_OUTPUT_DIR" \
    --k_best "$K_BEST" \
    --baseline

echo "Job finished."
