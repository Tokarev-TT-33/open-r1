# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from decimal import Decimal, InvalidOperation
from typing import Callable, Dict, Literal, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils.code_providers import get_provider
from .utils.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)
from .utils.competitive_programming import patch_code as cf_patch_code
from .utils.competitive_programming import score_submission as cf_score_submission
from .utils.competitive_programming import score_subtask


_THINK_BLOCK_RE = re.compile(r"<think>\n(.*?)\n</think>", flags=re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>\n(.*?)\n</answer>", flags=re.DOTALL)


def _extract_think_strict(text: str) -> str:
    """Extract <think>...</think> content. If missing, return empty string."""
    if not text:
        return ""
    m = _THINK_BLOCK_RE.search(text)
    return m.group(1).strip() if m else ""


def _extract_answer_strict(text: str) -> str:
    """Extract <answer>...</answer> content. If missing, return empty string."""
    if not text:
        return ""
    m = _ANSWER_BLOCK_RE.search(text)
    return m.group(1).strip() if m else ""


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # Allow optional whitespace between </think> and <answer> (models often emit an extra blank line).
    pattern = r"^<think>\n.*?\n</think>\s*<answer>\n.*?\n</answer>\s*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    # Guard: empty <think>/<answer> should get 0 to prevent "empty-content" reward hacking.
    rewards: list[float] = []
    for match, content in zip(matches, completion_contents):
        if not match:
            rewards.append(0.0)
            continue
        think = _extract_think_strict(content)
        answer = _extract_answer_strict(content)
        if not think or not answer:
            rewards.append(0.0)
            continue
        rewards.append(1.0)
    return rewards


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    # Only evaluate formatting within <think>...</think>
    think_contents = [_extract_think_strict(completion[0]["content"]) for completion in completions]
    matches = [len(re.findall(pattern, think)) for think in think_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def reasoning_similarity_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[float]:
    """
    Reward function that calculates the similarity between the reasoning process (<think> tag) 
    and the gold solution based on numerical and LaTeX key element overlap.
    
    This provides a denser reward signal by giving partial credit for intermediate steps 
    even if the final answer is incorrect.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Only compare within <think>...</think>
        think_content = _extract_think_strict(content)

        if not think_content or not sol:
            rewards.append(0.0)
            continue

        # Extract numbers and basic LaTeX elements (fractions, square roots)
        def extract_key_elements(text):
            # Matches integers, decimals, and basic LaTeX structures
            elements = re.findall(r"\d+\.?\d*|\\frac\{.*?\}\{.*?\}|\\sqrt\{.*?\}", text)
            return set(elements)

        model_elements = extract_key_elements(think_content)
        gold_elements = extract_key_elements(sol)

        if not gold_elements:
            rewards.append(0.0)
            continue

        # Calculate overlap (Jaccard-like recall)
        intersection = model_elements.intersection(gold_elements)
        recall = len(intersection) / len(gold_elements)

        # We cap this reward to provide a dense signal without overshadowing accuracy
        # Max reward of 0.5 for a perfect reasoning match
        rewards.append(recall * 0.5)

    return rewards


def reasoning_checkpoint_f1_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """
    Process reward that compares *verifiable checkpoints* (simple equations/assignments) in the model's <think>
    against checkpoints extracted from the gold `solution`.

    Key properties vs `reasoning_similarity_reward`:
      - Uses equation-like checkpoints (not raw numeric overlap)
      - Uses precision+recall (F1) to discourage "spam many equations"
      - Adds a weak order-consistency term (LIS over matched checkpoint indices)
      - Anchor-gates checkpoints to be relevant to the current problem/solution (reduces "1+1=2" spam)

    Output is always in [0, 1]. You can cap its contribution with `reward_weights` in config.
    """

    # Optional sympy-based equivalence check; we fall back to canonical string matching if unavailable.
    try:
        import sympy as sp  # type: ignore

        try:
            from sympy.parsing.latex import parse_latex as _parse_latex  # type: ignore
        except Exception:
            _parse_latex = None
    except Exception:
        sp = None  # type: ignore
        _parse_latex = None  # type: ignore

    def _extract_think(text: str) -> str:
        # Keep local helper, but enforce strict <think>\n...\n</think> extraction.
        return _extract_think_strict(text)

    def _normalize_text(s: str) -> str:
        s = s.strip()
        s = s.replace("\\,", "").replace("\\!", "")
        s = re.sub(r"\s+", " ", s)
        return s

    def _contains_nontrivial_symbol(s: str) -> bool:
        # Avoid rewarding pure numeric identities like "1+1=2"
        return bool(re.search(r"[A-Za-z]|\\[a-zA-Z]+", s))

    def _extract_candidate_lines(text: str, max_len: int = 220) -> list[str]:
        cands: list[str] = []
        for line in text.splitlines():
            line = _normalize_text(line)
            if not line or len(line) > max_len:
                continue
            if ("=" in line) or ("\\Rightarrow" in line) or ("\\implies" in line) or ("=>" in line):
                cands.append(line)
        return cands

    # Rough "=" split; conservative to avoid "=>", "<=", ">=".
    _eq_split = re.compile(r"(?<![<>])=(?!=)")

    _MATH_CHARS = r"A-Za-z0-9_\\\^\{\}\(\)\[\]\+\-\*/\.\s"

    def _trim_to_math_lhs(s: str) -> str:
        s = _normalize_text(s)
        # Take the trailing "math-like" chunk to drop prefixes like "We have".
        m = re.search(rf"([{_MATH_CHARS}]{{1,120}})$", s)
        return _normalize_text(m.group(1)) if m else s

    def _trim_to_math_rhs(s: str) -> str:
        s = _normalize_text(s)
        # Take the leading "math-like" chunk to drop suffix commentary.
        m = re.match(rf"^([{_MATH_CHARS}]{{1,120}})", s)
        return _normalize_text(m.group(1)) if m else s

    def _extract_equations_from_line(line: str) -> list[tuple[str, str]]:
        """
        Extract 1+ (lhs, rhs) equation candidates from a single line.
        Handles common patterns:
          - "x = 2"
          - "We have x = 2"
          - "y = x + 1 = 3" (chains)
        """
        line = _normalize_text(line)
        # First, split by common separators to reduce natural-language clutter.
        chunks = re.split(r"[;:。，,\.]", line)
        pairs: list[tuple[str, str]] = []

        for chunk in chunks:
            chunk = _normalize_text(chunk)
            if "=" not in chunk:
                continue
            parts = [p for p in _eq_split.split(chunk) if _normalize_text(p)]
            if len(parts) < 2:
                continue

            # Adjacent equalities: a=b, b=c ...
            for i in range(len(parts) - 1):
                lhs = _trim_to_math_lhs(parts[i])
                rhs = _trim_to_math_rhs(parts[i + 1])
                if lhs and rhs:
                    pairs.append((lhs, rhs))

            # Also add collapsed chain a=c for a=b=c (often closer to a "checkpoint").
            if len(parts) >= 3:
                lhs = _trim_to_math_lhs(parts[0])
                rhs = _trim_to_math_rhs(parts[-1])
                if lhs and rhs:
                    pairs.append((lhs, rhs))

        # De-dup preserving order.
        seen = set()
        uniq: list[tuple[str, str]] = []
        for lhs, rhs in pairs:
            key = _canonical_eq(lhs, rhs)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((lhs, rhs))
        return uniq

    def _anchors_from_gold_pairs(gold_pairs: list[tuple[str, str]]) -> set[str]:
        anchors: set[str] = set()
        for lhs, rhs in gold_pairs:
            s = f"{lhs} {rhs}"
            anchors |= set(re.findall(r"\b[A-Za-z]\b", s))  # single-letter vars
            anchors |= set(re.findall(r"\\[a-zA-Z]+", s))  # latex commands like \theta
        return anchors

    def _has_anchor(eq: str, anchors: set[str]) -> bool:
        # If anchors are empty (rare), don't gate.
        if not anchors:
            return True
        return any(a in eq for a in anchors)

    def _canonical_eq(lhs: str, rhs: str) -> str:
        # Cheap fallback for environments without LaTeX->sympy parsing.
        # We only try to normalize whitespace and trivial wrappers; this is intentionally simple.
        def _canon(s: str) -> str:
            s = _normalize_text(s)
            s = s.replace("{", "(").replace("}", ")")
            s = s.replace("\\left", "").replace("\\right", "")
            s = s.replace(" ", "")
            return s

        a, b = _canon(lhs), _canon(rhs)
        # Sort sides so "a=b" matches "b=a" in fallback mode.
        return "==".join(sorted([a, b]))

    def _try_parse_expr(tex: str):
        if _parse_latex is None:
            return None
        try:
            return _parse_latex(tex)
        except Exception:
            return None

    def _equiv_lr(lhs1: str, rhs1: str, lhs2: str, rhs2: str) -> bool:
        # Prefer sympy equivalence if available; else fallback to canonical string match.
        if sp is not None and _parse_latex is not None:
            a1, b1 = _try_parse_expr(lhs1), _try_parse_expr(rhs1)
            a2, b2 = _try_parse_expr(lhs2), _try_parse_expr(rhs2)
            if a1 is not None and b1 is not None and a2 is not None and b2 is not None:
                try:
                    def _eq(x, y) -> bool:
                        return bool(sp.simplify(x - y) == 0)

                    return (_eq(a1, a2) and _eq(b1, b2)) or (_eq(a1, b2) and _eq(b1, a2))
                except Exception:
                    pass
        return _canonical_eq(lhs1, rhs1) == _canonical_eq(lhs2, rhs2)

    def _lis_length(seq: list[int]) -> int:
        # O(n log n) LIS for order consistency score.
        import bisect

        d: list[int] = []
        for x in seq:
            i = bisect.bisect_left(d, x)
            if i == len(d):
                d.append(x)
            else:
                d[i] = x
        return len(d)

    contents = [completion[0]["content"] for completion in completions]
    rewards: list[float] = []

    for content, gold_sol in zip(contents, solution):
        think = _extract_think(content)
        if not think or not gold_sol:
            rewards.append(0.0)
            continue

        # 1) Extract gold checkpoints.
        gold_pairs: list[tuple[str, str]] = []
        for line in _extract_candidate_lines(gold_sol):
            if not _contains_nontrivial_symbol(line):
                continue
            gold_pairs.extend(_extract_equations_from_line(line))

        if not gold_pairs:
            rewards.append(0.0)
            continue

        anchors = _anchors_from_gold_pairs(gold_pairs)

        # 2) Extract model checkpoints (anchor-gated).
        model_pairs: list[tuple[str, str]] = []
        for line in _extract_candidate_lines(think):
            if not _contains_nontrivial_symbol(line):
                continue
            extracted = _extract_equations_from_line(line)
            for lhs, rhs in extracted:
                if not _has_anchor(f"{lhs} {rhs}", anchors):
                    continue
                model_pairs.append((lhs, rhs))

        if not model_pairs:
            rewards.append(0.0)
            continue

        # 3) Greedy match model checkpoints to gold checkpoints (one-to-one).
        matched_gold = [False] * len(gold_pairs)
        matched_gold_indices: list[int] = []
        matches = 0

        for (ml, mr) in model_pairs:
            found = -1
            for j, (gl, gr) in enumerate(gold_pairs):
                if matched_gold[j]:
                    continue
                if _equiv_lr(ml, mr, gl, gr):
                    found = j
                    break
            if found != -1:
                matched_gold[found] = True
                matched_gold_indices.append(found)
                matches += 1

        precision = matches / max(1, len(model_pairs))
        recall = matches / max(1, len(gold_pairs))
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Order score: fraction of matches that are in increasing gold order.
        if matches > 0:
            lis = _lis_length(matched_gold_indices)
            order_score = lis / max(1, matches)
        else:
            order_score = 0.0

        # Weighted combination, bounded in [0, 1].
        process_score = 0.7 * f1 + 0.3 * order_score
        rewards.append(float(min(1.0, max(0.0, process_score))))

    return rewards


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def cf_code_reward(
    completions,
    test_batch_size: int = 1,
    patch_code: bool = False,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    **kwargs,
) -> list[float]:
    """Reward function that evaluates Codeforces problems using Piston+our CF package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/codeforces (verifiable-prompts subset)

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
    code_snippets = [
        # note: grading is automatically skipped if a problem has no tests
        cf_patch_code(extract_code(completion[-1]["content"], language), language)
        if patch_code
        else extract_code(completion[-1]["content"], language)
        for completion, language in zip(completions, languages)
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return None

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                cf_score_submission(
                    piston_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                    scoring_mode=scoring_mode,
                    submission_language=problem_data.get("language", None),
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return results


def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    template = evaluation_script_template

    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    def code_format_reward(completions, **kwargs):
        # if there is a language field, use it instead of the default language. This way we can have mixed language training.
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(
                rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content, sample_language in zip(completion_contents, languages)
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


class ConsensusStepReward:
    """
    Anchor-Based Consensus reward.

    Within a single GRPO group, we:
      1) identify correct samples (by comparing completion final answer vs gold `solution`/`answer`)
      2) extract numeric "anchors" from correct samples
      3) declare "golden anchors" that appear in >= `consensus_threshold` fraction of correct samples
      4) reward each sample by the raw golden-anchor hit count (optionally capped)

    Notes:
      - This is designed for datasets that only contain `prompt` + final `solution` (no step labels).
      - Completions are assumed to be TRL-style chat completions: list[list[{"content": str, ...}]].
    """

    _NUM_RE = re.compile(r"-?\d+\.?\d*")

    def __init__(
        self,
        consensus_threshold: float = 0.6,
        min_correct: int = 2,
        max_golden_steps: Optional[int] = None,
    ):
        if not (0.0 <= consensus_threshold <= 1.0):
            raise ValueError(f"consensus_threshold must be in [0,1], got {consensus_threshold}")
        self.consensus_threshold = float(consensus_threshold)
        self.min_correct = int(min_correct)
        if self.min_correct < 1:
            raise ValueError(f"min_correct must be >= 1, got {self.min_correct}")
        self.max_golden_steps = None if max_golden_steps is None else int(max_golden_steps)
        if self.max_golden_steps is not None and self.max_golden_steps < 0:
            raise ValueError(f"max_golden_steps must be >= 0 (or None), got {self.max_golden_steps}")

        # TRL's GRPOTrainer collects reward function names via `reward_func.__name__`.
        # Instances of callable classes don't have `__name__` by default, so we provide one.
        self.__name__ = self.__class__.__name__

    def __call__(self, completions, **kwargs) -> list[float]:
        # 0) fetch gold answer list (aligned with completions)
        gold: Optional[list[str]] = kwargs.get("solution") or kwargs.get("answer")
        if gold is None:
            raise ValueError("ConsensusStepReward requires `solution` (or `answer`) in kwargs.")

        contents = [self._get_content(c) for c in completions]

        # 1) find correct sample indices
        correct_indices: list[int] = []
        for i, (content, sol) in enumerate(zip(contents, gold)):
            if self._is_correct(content, sol):
                correct_indices.append(i)

        # edge: require enough correct samples before we compute "golden" anchors
        if len(correct_indices) < self.min_correct:
            return [0.0] * len(contents)

        # 2) extract numeric anchor sets from correct samples
        anchors_per_correct: list[set[str]] = []
        for i in correct_indices:
            anchors_per_correct.append(self._extract_numbers_set(contents[i]))

        # 3) compute consensus and decide golden anchors
        freq: dict[str, int] = {}
        for s in anchors_per_correct:
            for a in s:
                freq[a] = freq.get(a, 0) + 1

        n_correct = len(anchors_per_correct)
        golden: set[str] = set()
        for a, c in freq.items():
            if (c / n_correct) >= self.consensus_threshold:
                golden.add(a)

        if not golden:
            return [0.0] * len(contents)

        # 4) score all samples by raw golden-anchor hit count (no normalization)
        rewards: list[float] = []
        for content in contents:
            sample_nums = self._extract_numbers_set(content)
            hits = len(sample_nums & golden)
            if self.max_golden_steps is not None:
                hits = min(hits, self.max_golden_steps)
            rewards.append(float(hits))
        return rewards

    @staticmethod
    def _get_content(completion) -> str:
        # TRL chat completion: completion[0]["content"]
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            return str(completion.get("content", ""))
        if isinstance(completion, list) and completion:
            item0 = completion[0]
            if isinstance(item0, dict):
                return str(item0.get("content", ""))
            if isinstance(item0, str):
                return item0
        return ""

    @staticmethod
    def _normalize_num_str(s: str) -> str:
        # Canonicalize numeric strings so "01", "1.0", "1." collapse to "1".
        try:
            d = Decimal(s)
        except (InvalidOperation, ValueError):
            return s
        d = d.normalize()
        out = format(d, "f").rstrip("0").rstrip(".") if "E" in str(d) or "e" in str(d) else str(d)
        if out in ("-0", "-0.0", "0.0"):
            out = "0"
        return out

    def _extract_numbers_set(self, text: str) -> set[str]:
        nums = self._NUM_RE.findall(text or "")
        return {self._normalize_num_str(n) for n in nums}

    @staticmethod
    def _is_correct(content: str, sol: str) -> bool:
        # Mirror `accuracy_reward` verification style (math_verify).
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
            if len(gold_parsed) == 0:
                # If gold is not parseable, fall back to a conservative string match.
                return sol.strip() != "" and sol.strip() in (content or "")

            answer_parsed = parse(
                content or "",
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            return bool(verify(gold_parsed, answer_parsed))
        except Exception:
            return False


class EmbeddingRubricReward:
    """
    AutoRubric-style *group-local* rubric reward (no global memory):

    - Within a single GRPO group (i.e., multiple generations for the same prompt), collect successful trajectories
      (final answer is correct), extract step-level checkpoints, then self-aggregate via embedding clustering to form
      "golden checkpoints".
    - Score each trajectory by how many golden checkpoints it overlaps with (optionally capped).

    This is designed to be domain-agnostic: math / coding / general reasoning.

    Backends:
      - "tfidf" (default): no extra model download; fast; weaker semantics.
      - "transformers": mean-pool last_hidden_state from a HF encoder model (must exist locally).
    """

    def __init__(
        self,
        # Mode:
        # - "group_rubric": AutoRubric-style group-local self-aggregation of golden checkpoints from successful trajectories.
        # - "pairwise": compare completion vs reference (solution/answer) with explicit <think>/<answer> separation.
        mode: Literal["group_rubric", "pairwise"] = "pairwise",
        rubric_threshold: float = 0.6,
        min_success: int = 2,
        merge_eps: float = 0.2,
        min_cluster_size: int = 2,
        match_threshold: float = 0.8,
        max_golden_hits: Optional[int] = None,
        # pairwise settings
        think_weight: float = 1.0,
        answer_weight: float = 1.0,
        pairwise_penalize_length_mismatch: bool = True,
        step_max_chars: int = 512,
        backend: Literal["tfidf", "transformers"] = "tfidf",
        embedding_model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.mode = mode
        if not (0.0 <= rubric_threshold <= 1.0):
            raise ValueError(f"rubric_threshold must be in [0,1], got {rubric_threshold}")
        self.rubric_threshold = float(rubric_threshold)
        self.min_success = int(min_success)
        if self.min_success < 1:
            raise ValueError(f"min_success must be >= 1, got {self.min_success}")

        # DBSCAN uses distance; we use cosine distance => eps in [0,2]. Smaller is stricter.
        self.merge_eps = float(merge_eps)
        if self.merge_eps <= 0:
            raise ValueError(f"merge_eps must be > 0, got {self.merge_eps}")
        self.min_cluster_size = int(min_cluster_size)
        if self.min_cluster_size < 1:
            raise ValueError(f"min_cluster_size must be >= 1, got {self.min_cluster_size}")

        self.match_threshold = float(match_threshold)
        if not (0.0 <= self.match_threshold <= 1.0):
            raise ValueError(f"match_threshold must be in [0,1], got {self.match_threshold}")

        self.max_golden_hits = None if max_golden_hits is None else int(max_golden_hits)
        if self.max_golden_hits is not None and self.max_golden_hits < 0:
            raise ValueError(f"max_golden_hits must be >= 0 (or None), got {self.max_golden_hits}")

        self.step_max_chars = int(step_max_chars)
        if self.step_max_chars < 32:
            raise ValueError(f"step_max_chars too small: {self.step_max_chars}")

        self.think_weight = float(think_weight)
        self.answer_weight = float(answer_weight)
        self.pairwise_penalize_length_mismatch = bool(pairwise_penalize_length_mismatch)

        self.backend = backend
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.device = device

        # TRL expects `reward_func.__name__`
        self.__name__ = self.__class__.__name__

        # lazy init
        self._tfidf: Optional[TfidfVectorizer] = None
        self._hf_tokenizer = None
        self._hf_model = None

    @staticmethod
    def _extract_think(text: str) -> str:
        # For completions we want strict tag-based extraction to avoid mixing <answer> into think scoring.
        # For references (e.g. dataset `solution`) there may be no tags; callers should pass raw text directly.
        return _extract_think_strict(text)

    @staticmethod
    def _extract_answer(text: str) -> str:
        return _extract_answer_strict(text)

    def _split_steps(self, completion_text: str) -> list[str]:
        think = self._extract_think(completion_text)
        # basic: split by non-empty lines
        raw_lines = [ln.strip() for ln in (think or "").splitlines()]
        steps = [ln for ln in raw_lines if ln]
        if not steps:
            return []
        # truncate each step to control compute
        return [s[: self.step_max_chars] for s in steps]

    def _init_hf(self):
        if self._hf_model is not None:
            return
        if not self.embedding_model_name_or_path:
            raise ValueError(
                "backend='transformers' requires embedding_model_name_or_path. "
                "You can leave it unset in YAML and we will default it to the training model's model_name_or_path in grpo.py."
            )
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name_or_path)
        self._hf_model = AutoModel.from_pretrained(self.embedding_model_name_or_path)
        if self.device:
            self._hf_model.to(self.device)
        self._hf_model.eval()
        self._torch = torch

    def _embed(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 1), dtype=np.float32)

        if self.backend == "tfidf":
            # NOTE: TF-IDF must be fitted once per group, otherwise vector dimensions won't match.
            # __call__ will set `self._tfidf` before calling this branch.
            if self._tfidf is None:
                raise RuntimeError("TF-IDF vectorizer is not initialized. This is a bug; __call__ should fit it per group.")
            X = self._tfidf.transform(texts)
            return X.toarray().astype(np.float32)

        if self.backend == "transformers":
            self._init_hf()
            tok = self._hf_tokenizer
            model = self._hf_model
            torch = self._torch
            with torch.no_grad():
                batch = tok(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                if self.device:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                out = model(**batch)
                last = out.last_hidden_state  # [B, T, H]
                mask = batch["attention_mask"].unsqueeze(-1).to(last.dtype)  # [B, T, 1]
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                emb = pooled.detach().cpu().numpy().astype(np.float32)
            # L2 normalize for cosine
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            return emb / norms

        raise ValueError(f"Unknown backend: {self.backend}")

    @staticmethod
    def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # A: [m, d], B: [n, d]
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
        # normalize (safe for tfidf dense)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return (An @ Bn.T).astype(np.float32)

    def __call__(self, completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] if isinstance(completion, list) else str(completion) for completion in completions]

        if self.mode == "pairwise":
            # Reference fields:
            # - Prefer `solution` as reference for <think> alignment (often long-form derivation).
            # - Prefer `answer` as reference for <answer> alignment (often final short answer).
            ref_think_list: Optional[list[str]] = kwargs.get("solution")
            ref_answer_list: Optional[list[str]] = kwargs.get("answer")
            if ref_think_list is None and ref_answer_list is None:
                raise ValueError("EmbeddingRubricReward(mode='pairwise') requires `solution` and/or `answer` in kwargs.")

            rewards: list[float] = []
            for i, content in enumerate(contents):
                gen_think_steps = self._split_steps(content)
                gen_answer = self._extract_answer(content)

                ref_think = ref_think_list[i] if ref_think_list is not None else ""
                ref_answer = ref_answer_list[i] if ref_answer_list is not None else ""
                # Reference solution typically has no <think> tags; use raw text as reference think.
                ref_think_steps = [ln.strip() for ln in (ref_think or "").splitlines() if ln.strip()]
                ref_think_steps = [s[: self.step_max_chars] for s in ref_think_steps]

                # Build a single TF-IDF vector space per sample for consistent dims (think+answer together is OK
                # because we never compare think vectors to answer vectors; they just share the same feature space).
                if self.backend == "tfidf":
                    texts_for_fit: list[str] = []
                    texts_for_fit.extend(gen_think_steps)
                    texts_for_fit.extend(ref_think_steps)
                    if gen_answer:
                        texts_for_fit.append(gen_answer[: self.step_max_chars])
                    if ref_answer:
                        texts_for_fit.append(ref_answer[: self.step_max_chars])
                    # if everything empty, reward is 0
                    if not texts_for_fit:
                        rewards.append(0.0)
                        continue
                    self._tfidf = TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=20000,
                    )
                    self._tfidf.fit(texts_for_fit)

                # think: one-to-one alignment by index
                think_score = 0.0
                if self.think_weight != 0.0:
                    n_gen = len(gen_think_steps)
                    n_ref = len(ref_think_steps)
                    if n_gen == 0 or n_ref == 0:
                        think_score = 0.0
                    else:
                        n = min(n_gen, n_ref)
                        A = self._embed(gen_think_steps[:n])
                        B = self._embed(ref_think_steps[:n])
                        sims = np.sum(A * B, axis=1) / (
                            (np.linalg.norm(A, axis=1) + 1e-12) * (np.linalg.norm(B, axis=1) + 1e-12)
                        )
                        # average, optionally penalize length mismatch by dividing by max(n_gen, n_ref)
                        denom = max(n_gen, n_ref) if self.pairwise_penalize_length_mismatch else n
                        think_score = float(np.clip(sims, -1.0, 1.0).sum() / max(1, denom))

                # answer: cosine similarity of answer spans
                answer_score = 0.0
                if self.answer_weight != 0.0:
                    if gen_answer and ref_answer:
                        A = self._embed([gen_answer[: self.step_max_chars]])
                        B = self._embed([ref_answer[: self.step_max_chars]])
                        sim = float((A @ B.T)[0, 0] / ((np.linalg.norm(A) + 1e-12) * (np.linalg.norm(B) + 1e-12)))
                        # map to [0,1] (optional): keep as [-1,1] could destabilize; for tfidf it is >=0 usually.
                        answer_score = max(0.0, sim)
                    else:
                        answer_score = 0.0

                total = self.think_weight * think_score + self.answer_weight * answer_score
                rewards.append(float(total))

            return rewards

        # Default: AutoRubric-style group-local self-aggregation
        # Need gold to determine success
        gold: Optional[list[str]] = kwargs.get("solution") or kwargs.get("answer")
        if gold is None:
            raise ValueError("EmbeddingRubricReward(mode='group_rubric') requires `solution` (or `answer`) in kwargs.")

        # identify successful trajectories
        success_idx: list[int] = []
        for i, (content, sol) in enumerate(zip(contents, gold)):
            if ConsensusStepReward._is_correct(content, sol):
                success_idx.append(i)

        if len(success_idx) < self.min_success:
            return [0.0] * len(contents)

        # Extract checkpoints (step texts) from successful trajectories (THINK ONLY)
        all_step_texts: list[str] = []
        step_owner: list[int] = []  # maps step -> which success trajectory (0..len(success_idx)-1)
        for owner_j, i in enumerate(success_idx):
            steps = self._split_steps(contents[i])
            for s in steps:
                all_step_texts.append(s)
                step_owner.append(owner_j)

        if len(all_step_texts) == 0:
            return [0.0] * len(contents)

        # TF-IDF: fit once on *all* think step texts in the group to keep a consistent vector space.
        if self.backend == "tfidf":
            group_step_texts: list[str] = []
            for content in contents:
                group_step_texts.extend(self._split_steps(content))
            if len(group_step_texts) == 0:
                return [0.0] * len(contents)
            self._tfidf = TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=20000,
            )
            self._tfidf.fit(group_step_texts)

        X = self._embed(all_step_texts)
        if X.shape[0] == 0:
            return [0.0] * len(contents)

        clustering = DBSCAN(eps=self.merge_eps, min_samples=self.min_cluster_size, metric="cosine")
        labels = clustering.fit_predict(X)

        clusters: dict[int, set[int]] = {}
        for step_i, lab in enumerate(labels):
            if lab == -1:
                continue
            clusters.setdefault(int(lab), set()).add(step_owner[step_i])

        if not clusters:
            return [0.0] * len(contents)

        n_success = len(success_idx)
        golden_cluster_ids: list[int] = []
        for cid, owners in clusters.items():
            if (len(owners) / n_success) >= self.rubric_threshold:
                golden_cluster_ids.append(cid)

        if not golden_cluster_ids:
            return [0.0] * len(contents)

        golden_centroids: list[np.ndarray] = []
        for cid in golden_cluster_ids:
            member_idx = np.where(labels == cid)[0]
            if member_idx.size == 0:
                continue
            centroid = X[member_idx].mean(axis=0, keepdims=False)
            golden_centroids.append(centroid.astype(np.float32))

        if not golden_centroids:
            return [0.0] * len(contents)

        G = np.stack(golden_centroids, axis=0)

        rewards: list[float] = []
        for content in contents:
            steps = self._split_steps(content)
            if not steps:
                rewards.append(0.0)
                continue
            Y = self._embed(steps)
            if Y.shape[0] == 0:
                rewards.append(0.0)
                continue
            sims = self._cosine_sim_matrix(Y, G)  # [num_steps, num_golden]
            matched = (sims.max(axis=0) >= self.match_threshold).sum()
            if self.max_golden_hits is not None:
                matched = min(int(matched), self.max_golden_hits)
            rewards.append(float(matched))

        return rewards


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "reasoning_similarity": reasoning_similarity_reward,
        "reasoning_checkpoint_f1": reasoning_checkpoint_f1_reward,
        "consensus": ConsensusStepReward(
            consensus_threshold=getattr(script_args, "consensus_threshold", 0.6),
            min_correct=getattr(script_args, "consensus_min_correct", 2),
            max_golden_steps=getattr(script_args, "consensus_max_golden_steps", None),
        ),
        "embedding_rubric": EmbeddingRubricReward(
            mode=getattr(script_args, "embedding_rubric_mode", "pairwise"),
            rubric_threshold=getattr(script_args, "embedding_rubric_threshold", 0.6),
            min_success=getattr(script_args, "embedding_rubric_min_success", 2),
            merge_eps=getattr(script_args, "embedding_rubric_merge_eps", 0.2),
            min_cluster_size=getattr(script_args, "embedding_rubric_min_cluster_size", 2),
            match_threshold=getattr(script_args, "embedding_rubric_match_threshold", 0.8),
            max_golden_hits=getattr(script_args, "embedding_rubric_max_golden_hits", None),
            think_weight=getattr(script_args, "embedding_rubric_think_weight", 1.0),
            answer_weight=getattr(script_args, "embedding_rubric_answer_weight", 1.0),
            pairwise_penalize_length_mismatch=getattr(
                script_args, "embedding_rubric_pairwise_penalize_length_mismatch", True
            ),
            step_max_chars=getattr(script_args, "embedding_rubric_step_max_chars", 512),
            backend=getattr(script_args, "embedding_rubric_backend", "tfidf"),
            embedding_model_name_or_path=getattr(script_args, "embedding_rubric_model_name_or_path", None),
            device=getattr(script_args, "embedding_rubric_device", None),
        ),
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "cf_code": update_wrapper(
            partial(
                cf_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                scoring_mode=script_args.code_eval_scoring_mode,
            ),
            cf_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
