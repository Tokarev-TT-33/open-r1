import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"

print(f"Loading from {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16, # Try float16
    device_map="auto",
    trust_remote_code=True
)

prompts = [
    "Hello",
    "Define a function f(x) = x^2 + 2x + 1. Find the value of f(3)."
]

for i, prompt in enumerate(prompts):
    print(f"\n--- Test {i+1}: '{prompt}' ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input shape: {inputs.input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        logits = outputs.logits
        
        print(f"Logits shape: {logits.shape}")
        print(f"Logits Mean: {logits.mean().item()}, Std: {logits.std().item()}")
        print(f"Logits Max: {logits.max().item()}, Min: {logits.min().item()}")
        
        if torch.isnan(logits).any():
            print("!!! NaN detected in logits !!!")
        else:
            print("Logits are clean.")
            
        # Try a simple greedy decoding manually
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        print(f"Next token ID: {next_token.item()}")
        print(f"Next token: {tokenizer.decode(next_token)}")

