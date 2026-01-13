#!/usr/bin/env bash
set -euo pipefail

# Self-Consistency (majority vote) evaluation for MATH-500 using vLLM.
#
# Examples:
#   # Quick debug
#   bash scripts/run_self_consistency_math500.sh --max-samples 20 --num-samples 8
#
#   # Full run (will take time!)
#   bash scripts/run_self_consistency_math500.sh --num-samples 8
#

MODEL="${MODEL:-/root/autodl-tmp/trained_models/Qwen3-4B-Math-220k-GRPO-full_v1/checkpoint-2300}"
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
SEED="${SEED:-1234}"
OUT_DIR="${OUT_DIR:-data/evals/self_consistency}"

source /root/open-r1/openr1/bin/activate

python scripts/self_consistency_math500.py \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  "$@"


