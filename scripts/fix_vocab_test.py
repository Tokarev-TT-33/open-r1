import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_path = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"

print("Loading Config...")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"Config Vocab Size: {config.vocab_size}")

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer Vocab Size: {tokenizer.vocab_size}")
print(f"Tokenizer Len: {len(tokenizer)}")

print("Loading Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print(f"Model Embedding Size: {model.get_input_embeddings().weight.shape}")
print(f"Model Head Size: {model.lm_head.weight.shape}")

# Try to fix the vocab mismatch manually and see if generation improves
target_vocab_size = len(tokenizer) # 151669

print(f"\nAttempting to fix vocab size to {target_vocab_size}...")

# 1. Resize input embeddings
model.resize_token_embeddings(target_vocab_size)
print(f"New Model Embedding Size: {model.get_input_embeddings().weight.shape}")

# 2. Resize output head (resize_token_embeddings usually handles this, but verify)
print(f"New Model Head Size: {model.lm_head.weight.shape}")

# 3. Test Generation
print("\nTesting generation after resize...")
prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(input_ids=inputs.input_ids)
    logits = outputs.logits
    print(f"Logits shape: {logits.shape}")
    print(f"Logits Mean: {logits.mean().item()}")
    print(f"Logits Std: {logits.std().item()}")
    print(f"Logits Max: {logits.max().item()}")
    print(f"Logits Min: {logits.min().item()}")

    # Generate
    gen_out = model.generate(inputs.input_ids, max_new_tokens=10, do_sample=False)
    print(f"Generated: {tokenizer.decode(gen_out[0])}")

