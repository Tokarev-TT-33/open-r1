import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"

# Load model without resizing first to see raw output
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Manually inspect the last few embeddings of the head
# Config says 151936, Tokenizer says 151669
# Let's see if the extra tokens (151669 -> 151936) are initialized or garbage
head_weight = model.lm_head.weight.data
print(f"Head shape: {head_weight.shape}")

# Check valid range (0 to 151669)
valid_weights = head_weight[:151669]
print(f"Valid range mean: {valid_weights.mean()}, std: {valid_weights.std()}")

# Check extra range (151669 to end)
extra_weights = head_weight[151669:]
print(f"Extra range mean: {extra_weights.mean()}, std: {extra_weights.std()}")

if extra_weights.mean() == 0 and extra_weights.std() == 0:
    print("Extra weights are all zero.")
else:
    print("Extra weights contain values.")

