import torch
import json
import time
import os
import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= Configuration =================
SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

TEST_PROMPTS = [
    "Define a function f(x) = x^2 + 2x + 1. Find the value of f(3).",
    "Solve the equation: 2x + 5 = 15.",
    "If a triangle has sides of length 3, 4, and 5, is it a right-angled triangle? Explain why.",
    "Calculate the integral of f(x) = x^2 from 0 to 3."
]


def load_model(model_path, lora_path=None):
    # 1. 简单的 Tokenizer 加载：如果 LoRA 目录下有 tokenizer，优先用它；否则用 base
    tokenizer_path = lora_path if lora_path and os.path.exists(os.path.join(lora_path, "tokenizer.json")) else model_path
    print(f"Loading tokenizer from {tokenizer_path}...")
    
    try:
        # 使用 fast tokenizer 通常更稳健
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # 仅做最基本的 pad_token 修正，防止报错
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    try:
        # 2. 稳健的模型加载：改用 float16，避免 bfloat16 可能的兼容性问题
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,  # 改回 float16 以求稳
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        try:
            # 3. 只要不报错，坚决不手动 resize，相信 PeftModel 的自动处理
            model = PeftModel.from_pretrained(model, lora_path)
            print("LoRA loaded successfully.")
        except Exception as e:
            # 如果这里报错说维度不匹配，那是真的不匹配，那时候再想办法
            print(f"FATAL: Failed to load LoRA: {e}")
            sys.exit(1)
            
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
    
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("\n[CRITICAL FAILURE] Model outputs contain NaN or Inf.")
        print("Possible causes: numerical overflow (bfloat16), corrupted weights, or token ID mismatch.")
        sys.exit(1)
    print("Sanity Check Passed.\n")

def format_prompt(tokenizer, prompt):
    # 4. 移除手动注入的模板逻辑，完全信任 tokenizer 自己的 apply_chat_template
    # 如果报错，说明这个 tokenizer 真的坏了或者不适合
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_benchmark(model, tokenizer, output_file):
    run_sanity_check(model, tokenizer)

    strategies = {
        "Greedy Search": {
            "do_sample": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "Top-P Sampling": {
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
        },
        "Top-K Sampling": {
            "do_sample": True,
            "top_k": 50,
            "temperature": 0.7,
        },
        "Beam Search": {
            "do_sample": False,
            "num_beams": 4,
            "early_stopping": True,
        },
    }

    results = []

    for prompt_text in TEST_PROMPTS:
        print(f"\n{'='*20}\nTesting Prompt: {prompt_text}\n{'='*20}")
        input_text = format_prompt(tokenizer, prompt_text)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        prompt_result = {"prompt": prompt_text, "outputs": {}}

        for strategy_name, strat_kwargs in strategies.items():
            print(f"Running {strategy_name}...", end="", flush=True)
            
            # 过滤不支持的参数
            current_kwargs = strat_kwargs.copy()
            if "top_k" in current_kwargs and not current_kwargs.get("do_sample", False):
                del current_kwargs["top_k"]
            if "top_p" in current_kwargs and not current_kwargs.get("do_sample", False):
                del current_kwargs["top_p"]
            if "temperature" in current_kwargs and not current_kwargs.get("do_sample", False):
                del current_kwargs["temperature"]

            gen_kwargs = {
                "max_new_tokens": 512,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                **current_kwargs
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

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run decoding comparison benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA adapter")
    parser.add_argument("--output_file", type=str, default="decoding_comparison_results.json", help="Output JSON file")
    
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        sys.exit(1)
        
    model, tokenizer = load_model(args.model_path, args.lora_path)
    run_benchmark(model, tokenizer, args.output_file)

if __name__ == "__main__":
    main()

