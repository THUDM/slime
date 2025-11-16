from __future__ import annotations

import re
from itertools import product

from math_verify import parse
from math_verify.grader import sympy_expr_eq
from sympy import Basic, MatrixBase

from .qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer


def extract_last_boxed(text: str) -> str | None:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(0)
    return None


def extract_solution(solution_str: str) -> tuple[str, bool]:
    model_output = re.sub(
        r"^.*?<\|im_start\|>assistant",
        "<|im_start|>assistant",
        solution_str,
        flags=re.DOTALL,
        count=1,
    )
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "[END]"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    predict_answer = qwen_extract_answer(model_output, data_name="math")
    boxed_answer = extract_last_boxed(model_output)
    return predict_answer, boxed_answer is not None


def verify_without_timeout(
    gold,
    target,
    float_rounding: int = 6,
    numeric_precision: int = 15,
    strict: bool = True,
) -> bool:
    from math_verify.utils import timeout

    @timeout(5)
    def compare_single_extraction(gold_item, target_item) -> bool:
        if isinstance(gold_item, (Basic, MatrixBase)) and isinstance(target_item, (Basic, MatrixBase)):
            return sympy_expr_eq(gold_item, target_item, float_rounding, numeric_precision, strict)
        if isinstance(gold_item, str) and isinstance(target_item, str):
            gold_item = gold_item.strip()
            target_item = target_item.strip()
            return len(gold_item) > 0 and gold_item == target_item
        return False

    def safe_compare(g, t):
        try:
            return compare_single_extraction(g, t)
        except Exception:
            return False

    if not isinstance(gold, list):
        gold = [gold]
    if not isinstance(target, list):
        target = [target]

    return any(safe_compare(g, t) for g, t in product(gold, target))


def hf_verify_with_try(gold, target) -> bool:
    try:
        parsed_target = parse(target)
        parsed_gold = parse(gold)
        return verify_without_timeout(gold=parsed_gold, target=parsed_target)
    except Exception as exc:  # noqa: BLE001
        print(f"Gold: {gold} Target: {target} Error: {exc}")
        return False


def compute_score(solution_str: str, ground_truth: str):
    extract_answer, is_boxed_matched = extract_solution(solution_str=solution_str)
    if "\\boxed" not in extract_answer:
        boxed_answer = f"\\boxed{{{extract_answer}}}"
    else:
        boxed_answer = extract_answer

    if "\\boxed" not in ground_truth:
        boxed_ground_truth = f"\\boxed{{{ground_truth}}}"
    else:
        boxed_ground_truth = ground_truth

    correct = hf_verify_with_try(gold=boxed_ground_truth, target=boxed_answer)
    answer_accuracy = 1 if correct else 0

    has_code_piece = re.search(
        r"```(?:py|python)?\n(.*?)\n```\nCode execution result:",
        solution_str,
        re.DOTALL,
    )

    return {
        "score": float(answer_accuracy),
        "extra_info": {
            "is_boxed_ratio": 1 if is_boxed_matched else 0,
            "score": float(answer_accuracy),
            "valid_code": 1 if has_code_piece else 0,
        },
    }
