import torch
import json
import time
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

# ================= Configuration =================
# 路径指向你本地已有的模型目录
# MODEL_PATH = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B" 
# BASE_MODEL_ID = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"

MODEL_PATH = "/root/autodl-tmp/trained_models/Qwen3-4B-Math-220k-GRPO-full_v1/checkpoint-2300" 
BASE_MODEL_ID = "Qwen3-4B-Math-220k-GRPO-full_v1"

# LoRA 模型路径 (建议先设为 None 跑通 Base 模型)
# LORA_PATH = '/root/open-r1/data/Qwen3-8B-Instruct-OpenR1-Math-SFT-LoRA' 
LORA_PATH = None

TEST_PROMPTS = [
    "Define a function f(x) = x^2 + 2x + 1. Find the value of f(3).",
    "Solve the equation: 2x + 5 = 15.",
    "If a triangle has sides of length 3, 4, and 5, is it a right-angled triangle? Explain why.",
    "Calculate the integral of f(x) = x^2 from 0 to 3."
]

SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

def load_model(model_path, base_model_id, lora_path=None):
    print(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    try:
        # Qwen 系列通常建议使用 bfloat16
        # 为了最大程度的兼容性，这里尝试 bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        
        # Explicitly check vocab size mismatch which is a common issue
        # Qwen models sometimes have mismatch between tokenizer vocab size and model embedding size
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            print(f"Warning: Vocab size mismatch! Model: {model.get_input_embeddings().weight.shape[0]}, Tokenizer: {len(tokenizer)}")
            print("Resizing model embeddings to match tokenizer...")
            model.resize_token_embeddings(len(tokenizer))
            
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        try:
            # resize_token_embeddings might be needed if LoRA adds new tokens
            # Checking vocab size match is important
            if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
                print(f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
                model.resize_token_embeddings(len(tokenizer))
            
            model = PeftModel.from_pretrained(model, lora_path)
        except Exception as e:
            print(f"Warning: Failed to load LoRA: {e}")
            print("Continuing with base model...")
    
    return model, tokenizer

def run_sanity_check(model, tokenizer):
    print("\nPerforming Model Sanity Check...")
    test_input = "Hello world"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids)
        logits = outputs.logits
        
    print(f"Logits Shape: {logits.shape}")
    print(f"Logits Mean: {logits.mean().item():.4f}")
    print(f"Logits Std:  {logits.std().item():.4f}")
    
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("\n[CRITICAL FAILURE] Model outputs contain NaN or Inf.")
        print("Possible causes: numerical overflow (bfloat16/float16), corrupted weights, or incompatible transformers version.")
        return False
        
    if logits.mean().item() == 0.0 and logits.std().item() == 0.0:
        print("\n[CRITICAL FAILURE] Model outputs are all ZEROs.")
        print("This indicates a broken model architecture implementation or corrupted weights.")
        print("The model is not computing anything valid.")
        return False
        
    print("Sanity Check Passed.\n")
    return True

def format_prompt(tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_benchmark(model, tokenizer):
    if not run_sanity_check(model, tokenizer):
        print("Aborting benchmark due to sanity check failure.")
        return

    strategies = {
        "Greedy Search": {
            "do_sample": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        # "Top-P Sampling": {
        #     "do_sample": True,
        #     "top_p": 0.9,
        #     "temperature": 0.7,
        # },
    }

    results = []

    for prompt_text in TEST_PROMPTS:
        print(f"\n{'='*20}\nTesting Prompt: {prompt_text}\n{'='*20}")
        input_text = format_prompt(tokenizer, prompt_text)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        prompt_result = {"prompt": prompt_text, "outputs": {}}

        for strategy_name, strat_kwargs in strategies.items():
            print(f"Running {strategy_name}...", end="", flush=True)
            
            gen_kwargs = {
                "max_new_tokens": 512,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                **strat_kwargs
            }
            
            try:
                start_time = time.time()
                with torch.no_grad():
                    output_ids = model.generate(**inputs, **gen_kwargs)
                
                generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                elapsed = time.time() - start_time
                print(f" Done. ({elapsed:.2f}s)")
                
                prompt_result["outputs"][strategy_name] = {"text": output_text, "time": elapsed}
                print(f"--- Output ({strategy_name}) ---")
                print(output_text[:200] + "..." if len(output_text) > 200 else output_text)
                print("-" * 30)
                
            except Exception as e:
                print(f" Failed. Error: {e}")
                prompt_result["outputs"][strategy_name] = {"error": str(e)}

        results.append(prompt_result)

    with open("decoding_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to decoding_comparison_results.json")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model path {MODEL_PATH} does not exist.")
        sys.exit(1)
        
    model, tokenizer = load_model(MODEL_PATH, BASE_MODEL_ID, lora_path=LORA_PATH)
    run_benchmark(model, tokenizer)
