import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/root/autodl-tmp/llm_models/Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 强制将 output head 的权重从 valid vocab size 之后截断
# 因为 Tokenizer 只有 151669，但模型配置是 151936
# 即使 resize_token_embeddings 也是重置大小，可能没有正确截断或处理
real_vocab_size = len(tokenizer)
print(f"Resizing to {real_vocab_size}")

# 方法1: 使用 resize_token_embeddings
model.resize_token_embeddings(real_vocab_size)

# 方法2: 强制检查 head 的权重是否正常
# 让我们打印一下 resize 后的最后几行的统计
head_weight = model.lm_head.weight.data
print(f"New head shape: {head_weight.shape}")
print(f"Last row mean: {head_weight[-1].mean()}, std: {head_weight[-1].std()}")

# 再跑一次前向传播
input_text = "Hello"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(input_ids=inputs.input_ids)
    logits = outputs.logits
    print(f"Logits mean: {logits.mean()}, std: {logits.std()}")
    
    # 检查是否有 NaN
    if torch.isnan(logits).any():
        print("Still NaN!")
    else:
        print("No NaN!")

    # Decode
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    print(f"Next token: {tokenizer.decode(next_token)}")

