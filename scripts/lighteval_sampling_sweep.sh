#!/usr/bin/env bash
set -euo pipefail

# 多采样策略 sweep（greedy / top-k / top-p / beam search），推荐使用 TP=2 双卡。
#
# 用法示例：
#   PARALLEL=tp NUM_GPUS=2 MODEL=/path/to/model TASK=math_500 bash scripts/lighteval_sampling_sweep.sh
#
# 快速 smoke test（只跑少量样本）：
#   PARALLEL=tp NUM_GPUS=2 bash scripts/lighteval_sampling_sweep.sh --max-samples 20
#
# 说明：
# - greedy/top-k/top-p：走 vLLM 的 generate() + SamplingParams
# - beam：走 vLLM 的 beam_search() + BeamSearchParams（需要我们已打的 lighteval patch）

MODE="${MODE:-compare}"
PARALLEL="${PARALLEL:-tp}"
NUM_GPUS="${NUM_GPUS:-2}"
MODEL="${MODEL:-/root/autodl-tmp/trained_models/Qwen3-4B-Math-220k-GRPO-full_v1/checkpoint-2300}"
TASK="${TASK:-math_500}"
OUT_BASE="${OUT_BASE:-data/evals}"

MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

# vLLM 稳定性参数（你之前遇到过 EngineCore_DP0 掉线/卡住，建议默认开 eager）
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"

# beam search 专用参数（强烈建议用这一套，否则很容易出现“显存很高但 GPU util 低 + 50s/token”的极慢现象）
# 你仍然可以通过环境变量覆盖这些默认值。
BEAM_WIDTH="${BEAM_WIDTH:-4}"
BEAM_MAX_NEW_TOKENS="${BEAM_MAX_NEW_TOKENS:-512}"
BEAM_MAX_MODEL_LENGTH="${BEAM_MAX_MODEL_LENGTH:-8192}"
BEAM_MAX_NUM_SEQS="${BEAM_MAX_NUM_SEQS:-8}"
BEAM_MAX_NUM_BATCHED_TOKENS="${BEAM_MAX_NUM_BATCHED_TOKENS:-16384}"
BEAM_ENFORCE_EAGER="${BEAM_ENFORCE_EAGER:-False}"

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# 透传额外的 lighteval CLI 参数，比如：--max-samples
EXTRA_ARGS="$@"

if [[ "$PARALLEL" != "tp" ]]; then
  echo "[ERROR] 这个 sweep 脚本默认只支持 PARALLEL=tp（避免 Ray/DP 相关卡住问题）。"
  echo "        你可以显式设置 PARALLEL=tp NUM_GPUS=2 再跑。"
  exit 1
fi

# 统一的 model args（除 generation_parameters 外）
MODEL_ARGS_PREFIX="model_name=$MODEL,dtype=bfloat16,enforce_eager=$ENFORCE_EAGER,max_num_seqs=$MAX_NUM_SEQS,max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=$GPU_MEM_UTIL,data_parallel_size=1,tensor_parallel_size=$NUM_GPUS"

timestamp="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(echo "$MODEL" | sed 's#/#_#g')"
RUN_DIR="$OUT_BASE/$MODEL_TAG/sweep/${TASK}_${timestamp}"

mkdir -p "$RUN_DIR"

run_one () {
  local name="$1"
  local gen="$2"
  local model_args_prefix="${3:-$MODEL_ARGS_PREFIX}"
  local out_dir="$RUN_DIR/$name"
  mkdir -p "$out_dir"

  echo
  echo "=============================="
  echo "[RUN] strategy=$name"
  echo "[OUT] $out_dir"
  echo "=============================="

  lighteval vllm \
    "$model_args_prefix,generation_parameters=$gen" \
    "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir "$out_dir" \
    $EXTRA_ARGS
}

# 注意：generation_parameters 是 {k:v,...} 的无引号 YAML/JSON-like 格式（lighteval 会转成 JSON 再解析）。
# NOTE:
# - 在 lighteval 的一些任务里，内部会请求 num_samples>1（比如 4），vLLM 会把 temperature==0 且 top_p==1/top_k==0
#   判定为“greedy”，并要求 n==1，导致报错。
# - 因此这里用一个极小的 temperature（1e-6）来近似 greedy，同时允许 n>1。
# 运行哪些策略：默认只跑 beam（保持你当前习惯，避免误触发全量对比）
# 如需跑全套对比：RUN_STRATEGIES="greedy top_p top_k beam"
RUN_STRATEGIES="${RUN_STRATEGIES:-beam}"

if [[ "$MODE" != "compare" && "$MODE" != "sanity" ]]; then
  echo "[ERROR] Unknown MODE=$MODE. Use MODE=compare or MODE=sanity."
  exit 1
fi

if [[ "$MODE" == "compare" ]]; then
  EFFECTIVE_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
  EFFECTIVE_MAX_MODEL_LENGTH="$MAX_MODEL_LENGTH"
else
  # sanity：快速试跑时建议整体缩短预算（你也可以自行覆盖 MAX_NEW_TOKENS/MAX_MODEL_LENGTH）
  EFFECTIVE_MAX_NEW_TOKENS="${SANITY_MAX_NEW_TOKENS:-512}"
  EFFECTIVE_MAX_MODEL_LENGTH="${SANITY_MAX_MODEL_LENGTH:-8192}"
fi

if [[ " $RUN_STRATEGIES " == *" greedy "* ]]; then
  run_one "greedy" "{max_new_tokens:$EFFECTIVE_MAX_NEW_TOKENS,seed:1234,temperature:1e-6,top_p:1.0,top_k:0}"
fi
if [[ " $RUN_STRATEGIES " == *" top_p "* ]]; then
  run_one "top_p"  "{max_new_tokens:$EFFECTIVE_MAX_NEW_TOKENS,seed:1234,temperature:0.7,top_p:0.9,top_k:0}"
fi
if [[ " $RUN_STRATEGIES " == *" top_k "* ]]; then
  run_one "top_k"  "{max_new_tokens:$EFFECTIVE_MAX_NEW_TOKENS,seed:1234,temperature:0.7,top_k:50,top_p:1.0}"
fi

# beam：单独用更合理的 vLLM 参数（默认禁用 eager 以启用 cudagraph；提高 max_num_batched_tokens；收紧 max_model_length/max_new_tokens）
if [[ " $RUN_STRATEGIES " == *" beam "* ]]; then
  # compare：对齐长度预算；sanity：用较短预算
  if [[ "$MODE" == "compare" ]]; then
    EFFECTIVE_BEAM_MAX_NEW_TOKENS="$EFFECTIVE_MAX_NEW_TOKENS"
    EFFECTIVE_BEAM_MAX_MODEL_LENGTH="$EFFECTIVE_MAX_MODEL_LENGTH"
  else
    EFFECTIVE_BEAM_MAX_NEW_TOKENS="$BEAM_MAX_NEW_TOKENS"
    EFFECTIVE_BEAM_MAX_MODEL_LENGTH="$BEAM_MAX_MODEL_LENGTH"
  fi

  BEAM_MODEL_ARGS_PREFIX="model_name=$MODEL,dtype=bfloat16,enforce_eager=$BEAM_ENFORCE_EAGER,max_num_seqs=$BEAM_MAX_NUM_SEQS,max_num_batched_tokens=$BEAM_MAX_NUM_BATCHED_TOKENS,max_model_length=$EFFECTIVE_BEAM_MAX_MODEL_LENGTH,gpu_memory_utilization=$GPU_MEM_UTIL,data_parallel_size=1,tensor_parallel_size=$NUM_GPUS"
  run_one "beam" "{max_new_tokens:$EFFECTIVE_BEAM_MAX_NEW_TOKENS,seed:1234,use_beam_search:true,beam_width:$BEAM_WIDTH,length_penalty:1.0,temperature:0.0}" "$BEAM_MODEL_ARGS_PREFIX"
fi

echo
echo "[DONE] sweep finished. Results under: $RUN_DIR"


