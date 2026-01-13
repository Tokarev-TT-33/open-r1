import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM

model_path = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"

print(f"Loading config from {model_path}")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

print(f"Original model_type: {config.model_type}")
# Force model_type to qwen2 if it is qwen3
if config.model_type == "qwen3":
    print("Forcing model_type to 'qwen2'...")
    config.model_type = "qwen2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model with modified config...")
# We load using AutoModelForCausalLM but passing the modified config
# If auto class mapping works for qwen2, it should load Qwen2ForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16, # Back to bfloat16 which is native
    device_map="auto",
    trust_remote_code=True
)

print(f"Loaded model class: {type(model)}")

prompt = "Define a function f(x) = x^2 + 2x + 1. Find the value of f(3)."
print(f"\n--- Test Prompt: '{prompt}' ---")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    logits = outputs.logits
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits Mean: {logits.mean().item()}, Std: {logits.std().item()}")
    
    if torch.isnan(logits).any():
        print("!!! NaN detected in logits !!!")
    else:
        print("Logits are clean.")
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        print(f"Next token: {tokenizer.decode(next_token)}")

