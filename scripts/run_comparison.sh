#!/bin/bash
set -e

# Activate environment if needed (assuming consistent with train_commands.sh)
source openr1/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Paths configured from user request
MODEL_PATH="/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"
LORA_PATH="/root/open-r1/data/Qwen3-8B-Instruct-OpenR1-Math-SFT-LoRA"
OUTPUT_FILE="decoding_comparison_results.json"

echo "Running decoding comparison..."
echo "Base Model: $MODEL_PATH"
echo "LoRA Path:  $LORA_PATH"

python3 scripts/evaluate_decoding.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --output_file "$OUTPUT_FILE"

echo "Done. Results saved to $OUTPUT_FILE"

