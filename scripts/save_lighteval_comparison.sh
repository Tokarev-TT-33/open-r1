#!/usr/bin/env bash
set -euo pipefail

# One-shot helper: generate a local comparison report (Markdown + CSV + JSON)
# from existing lighteval `results_*.json` files under data/evals.
#
# Usage:
#   bash scripts/save_lighteval_comparison.sh
#   FILTER_MODEL_CONTAINS=checkpoint-2300 bash scripts/save_lighteval_comparison.sh
#

SCAN_ROOT="${SCAN_ROOT:-data/evals}"
OUT_DIR="${OUT_DIR:-data/evals/comparisons}"
FILTER_TASK_CONTAINS="${FILTER_TASK_CONTAINS:-math_500}"
FILTER_MODEL_CONTAINS="${FILTER_MODEL_CONTAINS:-Qwen3-4B-Math-220k-GRPO-full_v1_checkpoint-2300}"
SORT_BY="${SORT_BY:-math_pass_1_1}"

python scripts/summarize_lighteval_results.py \
  --scan-root "$SCAN_ROOT" \
  --out-dir "$OUT_DIR" \
  --filter-task-contains "$FILTER_TASK_CONTAINS" \
  --filter-model-contains "$FILTER_MODEL_CONTAINS" \
  --sort-by "$SORT_BY"


