from .math_utils import extract_boxed_answer, grade_answer_sympy


def compute_score_geo3k(predict_str: str, ground_truth: str, use_boxed: bool = True, tol: float = 0.05) -> float:
    if use_boxed:
        answer = extract_boxed_answer(predict_str)
    else:
        answer = predict_str
    # Some geo3k examples have incorrect ground truth answers due to rounding to one or two decimal places.
    # For instance, the true answer might be 8/15, but the ground truth is given as 0.53.
    # The model might generate 8/15 (which is mathematically correct and can be parsed exactly), but this differs from 0.53.
    # To accommodate such cases, we use a tolerance when comparing the predicted answer to the ground truth.
    return 1.0 if grade_answer_sympy(answer, ground_truth, tol=tol) else 0.0
