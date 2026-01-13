import torch
from transformers import AutoModelForCausalLM

model_path = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"

print(f"Loading model from {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("\nChecking model weights statistics...")
total_params = 0
nonzero_params = 0

for name, param in model.named_parameters():
    if "embed_tokens" in name or "layers.0." in name or "lm_head" in name:
        data = param.data.float() # Convert to float for stat calculation
        mean = data.mean().item()
        std = data.std().item()
        min_val = data.min().item()
        max_val = data.max().item()
        is_zero = (data == 0).all().item()
        
        print(f"{name}: Mean={mean:.4f}, Std={std:.4f}, Min={min_val:.4f}, Max={max_val:.4f}, IsZero={is_zero}")

print("Done.")

