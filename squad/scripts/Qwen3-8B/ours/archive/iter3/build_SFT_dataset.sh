#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  echo "Error: could not determine REPO_ROOT" >&2
  exit 1
fi
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/squad"

SELF_EDIT_RESULTS_JSON_PATH=$REPO_ROOT/squad/results/Qwen3-8B/ours/archive/iter3_train/results.json
DATASET_OUTPUT_DIR=$REPO_ROOT/squad/data/Qwen3-8B/ours/archive/iter4
META_PROMPT_TXT_PATH=$REPO_ROOT/squad/data/meta_prompt_archive.txt
# This archive path is the path to the archive that was used to generate the current iter's data.
# It should be the same one that was passed to make_self_edits for the current iter.
ARCHIVE_PATH=$REPO_ROOT/squad/results/Qwen3-8B/ours/archive/iter2_train/archive.json
K_BEST=2

python3 -m "src.build_SFT_dataset" \
    --self_edit_results_json_path "$SELF_EDIT_RESULTS_JSON_PATH" \
    --dataset_output_dir "$DATASET_OUTPUT_DIR" \
    --meta_prompt_txt_path "$META_PROMPT_TXT_PATH" \
    --k_best "$K_BEST" \
    --archive_path "$archive_path"

echo "Job finished."
