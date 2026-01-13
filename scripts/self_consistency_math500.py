#!/usr/bin/env python3
"""
Self-Consistency evaluation for MATH-500 with vLLM.

What this script does:
- Load HuggingFaceH4/MATH-500 (test split)
- Build prompts using the same template as lighteval's `math_500` prompt
- Generate N sampled solutions per problem (top-p / top-k sampling)
- Extract math targets using lighteval's extraction utils (same as math_pass metrics)
- Self-consistency: take majority vote over extracted targets (best-effort), then score with sympy-based compare
- Also report:
  - single-sample pass@1 (using the first sample)
  - pass@1 with N samples (any correct among N) == pass@1:n_samples

Outputs:
- JSON with per-problem details + aggregate metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    Language,
    get_extraction_regexes,
    extract_target_from_pred,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target


MATH_QUERY_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.  The last line of your response should be of the "
    "following format: 'Therefore, the final answer is: $\\\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) "
    "where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.\n\n"
    "{Question}"
)


def build_prompt(question: str) -> str:
    return MATH_QUERY_TEMPLATE.format(Question=question).strip()


def apply_chat_template(tokenizer, user_text: str) -> str:
    # Match lighteval --use-chat-template behavior (no explicit system prompt by default).
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_extraction_regexes():
    return get_extraction_regexes(
        formatted_doc=None,
        target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
        language=Language.ENGLISH,
    )


def extract_targets(text: str, target_res) -> List[Any]:
    # lighteval's PassAtK uses any_match by default; keep identical defaults.
    return extract_target_from_pred(text, target_res)


def pick_majority_target(targets: List[Any]) -> Optional[Any]:
    """
    Majority vote over extracted targets.
    For sympy objects we vote by their string representation to be stable.
    """
    cleaned: List[Any] = [t for t in targets if t is not None and str(t).strip() != ""]
    if not cleaned:
        return None
    keys = [str(t) for t in cleaned]
    winner_key, _ = Counter(keys).most_common(1)[0]
    for t in cleaned:
        if str(t) == winner_key:
            return t
    return cleaned[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=8, help="Number of samples per problem (self-consistency k).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-samples", type=int, default=0, help="If >0, only evaluate this many problems (for quick debug).")
    ap.add_argument("--out-dir", type=str, default="data/evals/self_consistency")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"math500_self_consistency_{stamp}.json"

    t0 = time.time()

    ds = load_dataset("HuggingFaceH4/MATH-500", "default", split="test")
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    target_res = make_extraction_regexes()

    total = 0
    correct_any = 0
    correct_first = 0
    correct_sc = 0
    rows: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_generated_tokens = 0

    prompts: List[str] = []
    gold_solutions: List[str] = []
    for ex in ds:
        q = build_prompt(ex["problem"])
        prompts.append(apply_chat_template(tokenizer, q))
        gold_solutions.append(ex["solution"])

    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)

    for ex, gold, out in zip(ds, gold_solutions, outputs):
        total += 1
        # Token accounting (best-effort; useful for throughput / cost estimates)
        try:
            if getattr(out, "prompt_token_ids", None) is not None:
                total_prompt_tokens += len(out.prompt_token_ids)
        except Exception:
            pass
        pred_texts = [o.text for o in out.outputs]  # length == num_samples
        try:
            for o in out.outputs:
                if getattr(o, "token_ids", None) is not None:
                    total_generated_tokens += len(o.token_ids)
        except Exception:
            pass

        gold_targets = extract_targets(gold, target_res)
        pred_targets_per_sample = [extract_targets(t, target_res) for t in pred_texts]
        first_targets = pred_targets_per_sample[0] if pred_targets_per_sample else []

        # pass@1 with first sample
        is_first = bool(first_targets) and bool(gold_targets) and compare_gold_target(gold_targets, first_targets)
        # pass@1 with N samples (any correct)
        is_any = False
        for ts in pred_targets_per_sample:
            if ts and gold_targets and compare_gold_target(gold_targets, ts):
                is_any = True
                break

        # self-consistency majority vote (over first extracted target of each sample)
        flattened = []
        for ts in pred_targets_per_sample:
            if ts:
                flattened.append(ts[0])
        maj = pick_majority_target(flattened)
        is_sc = False
        if maj is not None and gold_targets:
            is_sc = compare_gold_target(gold_targets, [maj])

        correct_first += int(is_first)
        correct_any += int(is_any)
        correct_sc += int(is_sc)

        rows.append(
            {
                "id": ex.get("id"),
                "problem": ex.get("problem"),
                "gold_solution": gold,
                "gold_targets": [str(x) for x in (gold_targets or [])],
                "pred_texts": pred_texts,
                "pred_targets": [[str(x) for x in ts] for ts in pred_targets_per_sample],
                "sc_majority_target": str(maj) if maj is not None else None,
                "correct_first": bool(is_first),
                "correct_any": bool(is_any),
                "correct_self_consistency": bool(is_sc),
            }
        )

    summary = {
        "task": "math_500",
        "num_problems": total,
        "config": {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_samples": args.num_samples,
            "seed": args.seed,
        },
        "runtime": {
            "wall_time_seconds": None,  # filled below
            "total_prompt_tokens": int(total_prompt_tokens),
            "total_generated_tokens": int(total_generated_tokens),
        },
        "metrics": {
            "pass@1_first_sample": correct_first / max(1, total),
            "pass@1_any_of_k": correct_any / max(1, total),
            "self_consistency_majority_vote": correct_sc / max(1, total),
        },
    }

    t1 = time.time()
    summary["runtime"]["wall_time_seconds"] = float(t1 - t0)

    out_file.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote:", out_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


