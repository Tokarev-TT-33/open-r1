import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"
LORA_PATH = '/root/open-r1/data/Qwen3-8B-Instruct-OpenR1-Math-SFT-LoRA'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer len: {len(tokenizer)}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)
print(f"Model embeddings size: {model.get_input_embeddings().weight.shape}")
print(f"Model output size: {model.get_output_embeddings().weight.shape}")

if LORA_PATH:
    print("Loading LoRA...")
    model = PeftModel.from_pretrained(model, LORA_PATH)

print("Test forward pass...")
# input_text = "Hello"
input_text = "Define a function f(x) = x^2 + 2x + 1. Find the value of f(3)."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
print(f"Input IDs: {inputs.input_ids}")
print(f"Max Input ID: {inputs.input_ids.max()}")

with torch.no_grad():
    outputs = model(input_ids=inputs.input_ids)
    logits = outputs.logits
    print(f"Logits shape: {logits.shape}")
    print(f"Logits has NaN: {torch.isnan(logits).any()}")
    print(f"Logits has Inf: {torch.isinf(logits).any()}")
    print(f"Logits Mean: {logits.mean()}")
    print(f"Logits Std: {logits.std()}")

print("Done.")

