#!/usr/bin/env bash
set -euo pipefail

# 选择并行方式：
# - PARALLEL=none：单卡（默认，最稳）
# - PARALLEL=tp  ：tensor parallel，用 2 卡但不走 Ray（更稳，推荐）
# - PARALLEL=dp  ：data parallel，用 2 卡但会走 Ray（容易“卡住/报 metrics agent”，不推荐）
PARALLEL="${PARALLEL:-none}"

NUM_GPUS="${NUM_GPUS:-1}"
MODEL=/root/autodl-tmp/trained_models/Qwen3-4B-Math-220k-GRPO-full_v1/checkpoint-2300

# 默认用 vLLM generate（采样/近似 greedy），速度更正常；
# 如需 beam，请设置 STRATEGY=beam（会自动加 use_beam_search/beam_width 等参数，并启用更合理的 vLLM 配置）。
STRATEGY="${STRATEGY:-sample}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"

MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

# 默认 vLLM 参数（sample）
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"

# beam 专用更快配置（避免极慢）
BEAM_MAX_MODEL_LENGTH="${BEAM_MAX_MODEL_LENGTH:-8192}"
BEAM_MAX_NEW_TOKENS="${BEAM_MAX_NEW_TOKENS:-512}"
BEAM_ENFORCE_EAGER="${BEAM_ENFORCE_EAGER:-False}"
BEAM_MAX_NUM_SEQS="${BEAM_MAX_NUM_SEQS:-8}"
BEAM_MAX_NUM_BATCHED_TOKENS="${BEAM_MAX_NUM_BATCHED_TOKENS:-16384}"

if [[ "$STRATEGY" == "beam" ]]; then
  MODEL_ARGS_BASE="model_name=$MODEL,dtype=bfloat16,enforce_eager=$BEAM_ENFORCE_EAGER,max_num_seqs=$BEAM_MAX_NUM_SEQS,max_num_batched_tokens=$BEAM_MAX_NUM_BATCHED_TOKENS,max_model_length=$BEAM_MAX_MODEL_LENGTH,gpu_memory_utilization=$GPU_MEM_UTIL,generation_parameters={max_new_tokens:$BEAM_MAX_NEW_TOKENS,seed:1234,use_beam_search:true,beam_width:$BEAM_WIDTH,length_penalty:1.0,temperature:0.0}"
else
  MODEL_ARGS_BASE="model_name=$MODEL,dtype=bfloat16,enforce_eager=$ENFORCE_EAGER,max_num_seqs=$MAX_NUM_SEQS,max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=$GPU_MEM_UTIL,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:0.6,top_p:0.95}"
fi

if [[ "$PARALLEL" == "tp" ]]; then
  NUM_GPUS=2
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  MODEL_ARGS="$MODEL_ARGS_BASE,data_parallel_size=1,tensor_parallel_size=$NUM_GPUS"
elif [[ "$PARALLEL" == "dp" ]]; then
  NUM_GPUS=2
  # 走 Ray 的 data parallel：在容器里可能会出现 placement group / metrics agent 报错，看起来像卡住
  MODEL_ARGS="$MODEL_ARGS_BASE,data_parallel_size=$NUM_GPUS"
else
  # 单卡
  MODEL_ARGS="$MODEL_ARGS_BASE,data_parallel_size=1"
fi

TASK=math_500
OUTPUT_DIR=data/evals/$MODEL

# 允许把额外 CLI 参数透传给 lighteval，例如：
# bash scripts/lighteval_compare.sh --max-samples 10
EXTRA_ARGS="$@"

lighteval_args_banner () {
  echo
  echo "=============================="
  echo "[RUN] strategy=$STRATEGY"
  echo "[MODEL] $MODEL"
  echo "[OUT] $OUTPUT_DIR"
  echo "=============================="
}

lighteval_args_banner

lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    $EXTRA_ARGS


# export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
# MODEL=/root/autodl-tmp/trained_models/Qwen3-4B-Math-220k-GRPO-full_v1/checkpoint-2300
# MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# OUTPUT_DIR=data/evals/$MODEL

# # AIME 2024
# TASK=aime24
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# MATH-500
# TASK=math_500
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # GPQA Diamond
# TASK=gpqa:diamond
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # LiveCodeBench
# lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 