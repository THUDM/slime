# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from NVIDIA-NeMo/labs-molt at commit 9644c3f2cd9b:
# examples/python/utils/math_grader.py (the grader of molt's FlashREINFORCE recipe),
# with lint-only changes for slime's pre-commit (typing syntax, zip(strict=True)).

"""geo3k math-answer grader.

Two-tier answer equivalence:
  (1) string normalization (strip LaTeX wrappers, degrees, units, fix fracs/sqrt)
      then exact string equality;
  (2) a second normalization (degrees/units/latex-to-text/mixed-number) + tuple
      split + sympy symbolic equality, with strict rules so unreduced fractions
      and integers require an exact match (no symbolic simplification).

`sympy` and `pylatexenc` are optional: when absent, the symbolic / latex-to-text
tiers degrade gracefully and the string-normalization tiers still apply. Reward
is pure 0/1 accuracy.
"""

import re
from typing import Any


def _ground_truth_from_label(label: Any):
    if isinstance(label, dict):
        for key in ("ground_truth", "answer", "target", "label"):
            if key in label and label[key] not in (None, ""):
                return label[key]
        for key in ("reward_model", "extra_info"):
            if key in label:
                nested = _ground_truth_from_label(label[key])
                if nested not in (None, ""):
                    return nested
        solution = label.get("solution")
        if isinstance(solution, dict):
            for key in ("extracted", "answer", "ground_truth"):
                if key in solution and solution[key] not in (None, ""):
                    return solution[key]
    return label if label not in (None, "") else None


def _strip_prompt(query: str, prompt: Any) -> str:
    if isinstance(prompt, str):
        idx = query.rfind(prompt)
        if idx >= 0:
            return query[idx + len(prompt) :]
    return query


def _last_braced_command(text: str, command: str) -> str | None:
    matches = list(re.finditer(rf"{re.escape(command)}\s*{{", text))
    for match in reversed(matches):
        start = match.end()
        depth = 1
        chars = []
        for char in text[start:]:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars).strip()
            chars.append(char)
    return None


def extract_answer(text: str) -> str | None:
    """Extract the final answer: prefer the last balanced \\boxed{}/\\fbox{},
    then an "answer is ..." phrase, then the last number."""
    if not text:
        return None

    for command in (r"\boxed", r"\fbox"):
        boxed = _last_braced_command(text, command)
        if boxed:
            return boxed

    answer_matches = list(
        re.finditer(
            r"(?:final\s+answer|answer|therefore|thus)\s*(?:is|:|=)?\s*([^\n<|]+)",
            text,
            flags=re.IGNORECASE,
        )
    )
    for match in reversed(answer_matches):
        candidate = match.group(1).strip()
        if candidate:
            return candidate

    numeric_matches = re.findall(
        r"(?:[-+]?\d+(?:\.\d+)?\s*%?)|(?:\\frac\s*{[^{}]+}\s*{[^{}]+})",
        text,
    )
    if numeric_matches:
        return numeric_matches[-1].strip()

    return None


# ======================================================================
# Tier 1: string normalization + exact equality.
# ======================================================================
def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string: str) -> str:
    # "\\text{ " marks a trailing unit description.
    if "\\text{ " in string:
        return string.split("\\text{ ")[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("°", "")  # unicode degree symbol
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    # drop a short leading assignment, e.g. "k = ".
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _normalize_str(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


# ======================================================================
# Tier 2: degrees/units/latex normalization + sympy symbolic equality.
# ======================================================================
_BAD_SUBSTRINGS = ["^{", "^("]
_BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
_TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    from sympy.parsing import sympy_parser

    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    from pylatexenc import latex2text

    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # play nice with mixed numbers
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str) -> str:
    # e.g. "7 3/4" => "7+3/4"
    return re.compile("([0-9]) +([0-9])").sub("\\1+\\2", step)


def _strip_properly_formatted_commas(expr: str) -> str:
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return expr


def _normalize(expr: str | None) -> str | None:
    if expr is None:
        return None
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")
    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)
    expr = expr.replace("°", "")  # unicode degree symbol
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def _count_unknown_letters_in_expr(expr: str) -> int:
    expr = expr.replace("sqrt", "").replace("frac", "")
    return len({x for x in expr if x.isalpha()})


def _should_allow_eval(expr: str) -> bool:
    if _count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in _BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in _BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def _are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    try:
        import sympy
    except Exception:
        return False
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if _should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            if sympy.simplify(sympy_diff) == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def _split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in _TUPLE_CHARS
        and expr[-1] in _TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in _TUPLE_CHARS)
    ):
        return [elem.strip() for elem in expr[1:-1].split(",")]
    return [expr]


def grade_answer(given_answer: str | None, ground_truth: str | None) -> bool:
    """True if the normalized strings match OR sympy simplifies their difference
    to 0. Unreduced fractions and integers require an exact (non-symbolic) match."""
    if given_answer is None or ground_truth is None:
        return False

    gt_norm1 = _normalize_str(ground_truth)
    given_norm1 = _normalize_str(given_answer)
    if gt_norm1 is not None and gt_norm1 == given_norm1:
        return True

    gt_norm = _normalize(ground_truth)
    given_norm = _normalize(given_answer)
    if gt_norm is None:
        return False
    if gt_norm == given_norm:
        return True
    if given_norm is None or len(given_norm) == 0:
        return False

    gt_elems = _split_tuple(gt_norm)
    given_elems = _split_tuple(given_norm)
    if len(gt_elems) > 1 and (gt_norm[0] != given_norm[0] or gt_norm[-1] != given_norm[-1]):
        return False
    if len(gt_elems) != len(given_elems):
        return False

    is_correct = False
    for gt_elem, given_elem in zip(gt_elems, given_elems, strict=True):
        if _is_frac(gt_elem) and _is_frac(given_elem):
            # unreduced fractions must match exactly (no symbolic simplification)
            is_correct = gt_elem == given_elem
        elif _str_is_int(gt_elem) != _str_is_int(given_elem):
            # an integer ground truth requires a strict (non-symbolic) match
            is_correct = False
        else:
            is_correct = _are_equal_under_sympy(gt_elem, given_elem)
        if not is_correct:
            break
    return is_correct


def normalize_answer(answer: Any) -> str:
    """Tier-1 string normalization (public, backward-compatible)."""
    return _normalize_str(str(answer)) or ""


def answers_match(prediction: Any, target: Any) -> bool:
    if prediction in (None, "") or target in (None, ""):
        return False
    return grade_answer(str(prediction), str(target))


def score_response(query: str, prompt: Any, label: Any) -> dict[str, Any]:
    target = _ground_truth_from_label(label)
    response = _strip_prompt(query, prompt)
    prediction = extract_answer(response) or extract_answer(query)
    correct = answers_match(prediction, target)
    return {
        "reward": 1.0 if correct else 0.0,
        "prediction": prediction or "",
        "target": target or "",
        "missing_answer": 0.0 if prediction else 1.0,
        "missing_target": 0.0 if target not in (None, "") else 1.0,
    }
