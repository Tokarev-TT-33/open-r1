#!/usr/bin/env bash
set -euo pipefail

# 单独跑 beam search（vLLM beam_search + BeamSearchParams）
# 目标：避免极慢配置（eager + 超小 max_num_batched_tokens + 超长 max_new_tokens/max_model_length）
#
# 用法示例：
#   PARALLEL=tp NUM_GPUS=2 MODEL=/path/to/model TASK=math_500 bash scripts/lighteval_beam.sh --max-samples 20
#

MODE="${MODE:-compare}"
PARALLEL="${PARALLEL:-tp}"
NUM_GPUS="${NUM_GPUS:-2}"
MODEL="${MODEL:-/root/autodl-tmp/trained_models/Qwen3-4B-Math-220k-GRPO-full_v1/checkpoint-2300}"
TASK="${TASK:-math_500}"
OUT_BASE="${OUT_BASE:-data/evals}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"

# 统一预算参数（用于 MODE=compare，可与其它采样策略严格对齐）
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"

# beam 的默认“快跑”预算（用于 MODE=sanity，保证 beam 可用、快速）
BEAM_MAX_NEW_TOKENS="${BEAM_MAX_NEW_TOKENS:-512}"
BEAM_MAX_MODEL_LENGTH="${BEAM_MAX_MODEL_LENGTH:-8192}"

BEAM_ENFORCE_EAGER="${BEAM_ENFORCE_EAGER:-False}"
BEAM_MAX_NUM_SEQS="${BEAM_MAX_NUM_SEQS:-8}"
BEAM_MAX_NUM_BATCHED_TOKENS="${BEAM_MAX_NUM_BATCHED_TOKENS:-16384}"

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ "$PARALLEL" != "tp" ]]; then
  echo "[ERROR] 该脚本只支持 PARALLEL=tp（避免 Ray/DP 相关卡住问题）。"
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(echo "$MODEL" | sed 's#/#_#g')"
OUT_DIR="$OUT_BASE/$MODEL_TAG/beam/${TASK}_${timestamp}"
mkdir -p "$OUT_DIR"

if [[ "$MODE" == "compare" ]]; then
  EFFECTIVE_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
  EFFECTIVE_MAX_MODEL_LENGTH="$MAX_MODEL_LENGTH"
elif [[ "$MODE" == "sanity" ]]; then
  EFFECTIVE_MAX_NEW_TOKENS="$BEAM_MAX_NEW_TOKENS"
  EFFECTIVE_MAX_MODEL_LENGTH="$BEAM_MAX_MODEL_LENGTH"
else
  echo "[ERROR] Unknown MODE=$MODE. Use MODE=compare or MODE=sanity."
  exit 1
fi

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,enforce_eager=$BEAM_ENFORCE_EAGER,max_num_seqs=$BEAM_MAX_NUM_SEQS,max_num_batched_tokens=$BEAM_MAX_NUM_BATCHED_TOKENS,max_model_length=$EFFECTIVE_MAX_MODEL_LENGTH,gpu_memory_utilization=$GPU_MEM_UTIL,data_parallel_size=1,tensor_parallel_size=$NUM_GPUS"
GEN_ARGS="{max_new_tokens:$EFFECTIVE_MAX_NEW_TOKENS,seed:1234,use_beam_search:true,beam_width:$BEAM_WIDTH,length_penalty:1.0,temperature:0.0}"

echo
echo "=============================="
echo "[RUN] strategy=beam"
echo "[MODE]  $MODE"
echo "[MODEL] $MODEL"
echo "[TASK]  $TASK"
echo "[OUT]   $OUT_DIR"
echo "[ARGS]  width=$BEAM_WIDTH max_new_tokens=$EFFECTIVE_MAX_NEW_TOKENS max_model_length=$EFFECTIVE_MAX_MODEL_LENGTH"
echo "=============================="

lighteval vllm \
  "$MODEL_ARGS,generation_parameters=$GEN_ARGS" \
  "lighteval|$TASK|0|0" \
  --use-chat-template \
  --output-dir "$OUT_DIR" \
  "$@"


