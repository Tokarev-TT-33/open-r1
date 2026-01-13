# ==========================================
# Open-R1 训练启动命令备忘录
# ==========================================

# 注意：每次运行前，请确保已经清理了可能残留的进程
# pkill -f python

# ------------------------------------------
# 1. SFT 全量微调 (Full Fine-tuning)
# ------------------------------------------
# 适用场景：追求极致效果，且显存充足（使用 ZeRO-3）
# 配置文件：recipes/Qwen3-8B/sft/config_math.yaml
# 特点：ZeRO-3, Batch Size=2, Gradient Accumulation=16, LR=1e-5
# (修改了 GradAcc=16 以获得 GlobalBS=64, LR降为1e-5以防止Loss发散)

source openr1/bin/activate

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

deepspeed --num_gpus=2 src/open_r1/sft.py --config recipes/Qwen3-8B/sft/config_math.yaml


# ------------------------------------------
# 2. SFT LoRA 微调 (Low-Rank Adaptation)
# ------------------------------------------
# 适用场景：追求速度，显存有限，快速实验
# 配置文件：recipes/Qwen3-8B/sft/config_math_lora.yaml
# 特点：ZeRO-2, Batch Size=8, Gradient Checkpointing=ON
# (注意：如果追求极致速度，可以把 config 中的 gradient_checkpointing 设为 false 并加大 BS)

source openr1/bin/activate

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

deepspeed --num_gpus=2 src/open_r1/sft.py --config recipes/Qwen3-8B/sft/config_math_lora.yaml


# ------------------------------------------
# 3. GRPO 训练 (Reinforcement Learning)
# ------------------------------------------
# 配置文件：recipes/Qwen3-4B/grpo/config_demo.yaml
# 已设置 WANDB_MODE=offline 以防止断连问题

source openr1/bin/activate

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -rf /root/autodl-tmp/cache/*

# 使用 accelerate launch 并开启 --vllm_mode colocate
# 显存优化：通过 vllm_gpu_memory_utilization 控制 vLLM 占用
# 默认已开启 resume_from_checkpoint: true
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    src/open_r1/grpo.py \
    --config recipes/Qwen3-4B/grpo/config_demo.yaml \
    --vllm_mode colocate

# ------------------------------------------
# ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --config_file recipes/accelerate_configs/ddp.yaml \
#     src/open_r1/grpo.py \
#     --config recipes/Qwen3-4B/grpo/config_demo.yaml \
#     --vllm_mode colocate
# ------------------------------------------
# 4. 训练后评估 (Evaluation)
# ------------------------------------------
# 训练完成后，使用此命令测试模型在数学数据集上的真实准确率
# 这将运行 math_500 benchmark

source openr1/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 替换 --model_id 为你的实际输出路径
python3 scripts/run_benchmarks.py \
    --model_id /root/autodl-tmp/trained_models/Qwen3-8B-Instruct-OpenR1-Math-SFT \
    --benchmarks math_500 \
    --trust_remote_code

# ------------------------------------------
# 常见问题排查
# ------------------------------------------
# 1. 如果报错 "Address already in use" 或卡住：
#    执行：pkill -f python
#
# 2. 如果报错 "No space left on device"：
#    执行：rm -rf /root/autodl-tmp/cache/*
#
# 3. 如果报错 Flash Attention Padding Side 问题：
#    确保本地模型目录下的 tokenizer_config.json 中包含 "padding_side": "left"
#
# 同步数据：
# while true; do wandb sync wandb/latest-run; sleep 600; done
