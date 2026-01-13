#!/usr/bin/env bash
set -euo pipefail

# Compute-budget comparison runner for MATH-500.
#
# Why this exists
# - Beam search (beam_width=4) is extremely slow with max_new_tokens=2048 (hours).
# - "Fair" comparison can be defined in two ways:
#   A) Same decoding budget (same max_new_tokens/max_model_length)  -> beam is slow but comparable.
#   B) Same compute budget ("同算力预算"): compare beam_width=4 vs sampling with n=4 candidates
#      (roughly similar forward-pass count per token-step), and optionally self-consistency majority vote.
#
# This script provides BOTH, without removing your original 8h plan.
#
# Usage
#   # 8h+ run (same-budget beam, width=4)
#   bash scripts/run_compute_budget_compare.sh beam_full
#
#   # Compute-budget run (sampling n=4 + self-consistency k=4)
#   bash scripts/run_compute_budget_compare.sh compute_budget
#
#   # Quick debug
#   bash scripts/run_compute_budget_compare.sh compute_budget --max-samples 20
#

CMD="${1:-}"
shift || true

if [[ -z "$CMD" || "$CMD" == "-h" || "$CMD" == "--help" ]]; then
  echo "Usage: $0 {beam_full|compute_budget} [extra lighteval args, e.g. --max-samples 20]"
  exit 0
fi

source /root/open-r1/openr1/bin/activate

MODEL="${MODEL:-/root/autodl-tmp/trained_models/Qwen3-4B-Math-220k-GRPO-full_v1/checkpoint-2300}"
TASK="${TASK:-math_500}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
SEED="${SEED:-1234}"

# vLLM performance knobs (should NOT change decoding semantics)
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

timestamp="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(echo "$MODEL" | sed 's#/#_#g')"

MODEL_ARGS_PREFIX="model_name=$MODEL,dtype=bfloat16,enforce_eager=$ENFORCE_EAGER,max_num_seqs=$MAX_NUM_SEQS,max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=$GPU_MEM_UTIL,data_parallel_size=1,tensor_parallel_size=$NUM_GPUS"

if [[ "$CMD" == "beam_full" ]]; then
  # Ensure beam bugfix applied (safe no-op if already patched)
  bash scripts/apply_lighteval_beam_fix.sh || true

  echo
  echo "=============================="
  echo "[RUN] beam_full (same decoding budget)"
  echo "[MODEL] $MODEL"
  echo "[TASK]  $TASK"
  echo "[NOTE]  This is the long 8h-class run: beam_width=4, max_new_tokens=$MAX_NEW_TOKENS"
  echo "=============================="

  MODE=compare PARALLEL=tp NUM_GPUS="$NUM_GPUS" TASK="$TASK" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" MAX_MODEL_LENGTH="$MAX_MODEL_LENGTH" \
    BEAM_WIDTH="${BEAM_WIDTH:-4}" \
    BEAM_MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" BEAM_MAX_NUM_SEQS="${BEAM_MAX_NUM_SEQS:-8}" BEAM_ENFORCE_EAGER="$ENFORCE_EAGER" \
    bash scripts/lighteval_beam.sh "$@"

  echo
  echo "[DONE] beam_full finished."
  echo "Tip: regenerate comparison report:"
  echo "  bash scripts/save_lighteval_comparison.sh"
  exit 0
fi

if [[ "$CMD" == "compute_budget" ]]; then
  # Compute-budget: compare beam_width=4 vs sampling with n=4 candidates.
  # Here we run sampling only (fast). Beam baseline is still available via `beam_full`.
  #
  # IMPORTANT:
  # - In this lighteval version, `generation_parameters.n` is NOT an allowed config field (pydantic will error).
  # - For `math_500`, lighteval will automatically request the needed number of samples for metrics; since
  #   the task includes `math_pass@1:4_samples`, the backend will generate 4 samples when needed.
  # - lighteval's math_pass@1:4_samples is PassAtK(k=1,n=4): "any-of-4" correctness, NOT majority vote.
  # - We additionally run our self-consistency script (majority vote) for a true SC metric.
  N_CANDIDATES="${N_CANDIDATES:-4}"
  OUT_BASE="${OUT_BASE:-data/evals}"
  RUN_DIR="$OUT_BASE/$MODEL_TAG/compute_budget/${TASK}_${timestamp}"

  mkdir -p "$RUN_DIR"
  echo
  echo "=============================="
  echo "[RUN] compute_budget (same compute-ish budget)"
  echo "[MODEL] $MODEL"
  echo "[TASK]  $TASK"
  echo "[OUT]   $RUN_DIR"
  echo "[BUDGET] sampling n=$N_CANDIDATES (approx ~ beam_width=$N_CANDIDATES)"
  if [[ "$TASK" == "math_500" && "$N_CANDIDATES" != "4" ]]; then
    echo "[WARN] lighteval math_500 metrics are hard-coded to n=4 (math_pass@1:4_samples)."
    echo "       We'll still run lighteval (n=4 implicit), but self-consistency will use NUM_SAMPLES=$N_CANDIDATES."
  fi
  echo "=============================="

  # Greedy-ish with n>1 (lighteval/vLLM requires non-zero temperature when n>1)
  lighteval vllm \
    "$MODEL_ARGS_PREFIX,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,seed:$SEED,temperature:1e-6,top_p:1.0,top_k:0}" \
    "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir "$RUN_DIR/greedy_n${N_CANDIDATES}" \
    "$@"

  # Top-p sampling with n candidates (for any-of-k pass@1:n and for self-consistency)
  TEMP="${TEMP:-0.7}"
  TOP_P="${TOP_P:-0.9}"
  lighteval vllm \
    "$MODEL_ARGS_PREFIX,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,seed:$SEED,temperature:$TEMP,top_p:$TOP_P,top_k:0}" \
    "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir "$RUN_DIR/top_p_n${N_CANDIDATES}" \
    "$@"

  # Optional: top-k
  TOP_K="${TOP_K:-50}"
  lighteval vllm \
    "$MODEL_ARGS_PREFIX,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,seed:$SEED,temperature:$TEMP,top_k:$TOP_K,top_p:1.0}" \
    "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir "$RUN_DIR/top_k_n${N_CANDIDATES}" \
    "$@"

  echo
  echo "=============================="
  echo "[RUN] self-consistency majority vote (k=$N_CANDIDATES)"
  echo "=============================="
  NUM_SAMPLES="$N_CANDIDATES" TEMP="$TEMP" TOP_P="$TOP_P" TOP_K=0 \
    MODEL="$MODEL" TP="$NUM_GPUS" MAX_MODEL_LEN="$MAX_MODEL_LENGTH" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" SEED="$SEED" \
    OUT_DIR="$RUN_DIR/self_consistency" \
    bash scripts/run_self_consistency_math500.sh "$@"

  echo
  echo "[DONE] compute_budget finished. Regenerating comparison report..."
  bash scripts/save_lighteval_comparison.sh || true
  echo
  echo "[DONE] Reports under:"
  echo "  - $RUN_DIR"
  echo "  - data/evals/comparisons/"
  exit 0
fi

echo "[ERROR] unknown subcommand: $CMD"
exit 2


