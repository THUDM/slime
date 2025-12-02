import re

from .math_utils import extract_boxed_answer, grade_answer_sympy


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_answer(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer_sympy(answer, ground_truth, tol=0.05) else 0.0


def compute_score_geo3k(
    predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.0
) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )
