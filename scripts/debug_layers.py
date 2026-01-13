import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"Input IDs: {inputs.input_ids}")

with torch.no_grad():
    # 1. Embeddings
    print("\nChecking Embeddings...")
    embeds = model.get_input_embeddings()(inputs.input_ids)
    print(f"Embeddings: Mean={embeds.mean()}, Std={embeds.std()}, Max={embeds.max()}, Min={embeds.min()}")
    
    # 2. Layer 0
    print("\nChecking Layer 0...")
    hidden_states = embeds
    layer0 = model.model.layers[0]
    
    # Forward through Layer 0 manually (simplified)
    # Note: This misses position_ids and proper attention mask, but basic check
    try:
        output0 = layer0(hidden_states, attention_mask=None, position_ids=None)[0]
        print(f"Layer 0 Output: Mean={output0.mean()}, Std={output0.std()}")
    except Exception as e:
        print(f"Layer 0 forward failed: {e}")

    # 3. Full forward
    print("\nChecking Full Forward...")
    outputs = model(input_ids=inputs.input_ids)
    logits = outputs.logits
    print(f"Final Logits: Mean={logits.mean()}, Std={logits.std()}")

