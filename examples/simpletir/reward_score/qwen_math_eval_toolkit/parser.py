import re
from typing import Any, Dict

import regex
from word2number import w2n


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
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
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        return f"\\frac{{{a}}}{{{b}}}"
    except Exception:
        return string


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except Exception:
        pass
    return text


unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
]


def clean_float_text(string: str) -> str:
    string = convert_word_number(string)
    string = string.replace(",", "")
    string = string.replace(" ", "")
    string = string.replace("\\mathit", "")
    string = string.replace("\\textit", "")
    string = string.replace("\\text", "")
    string = string.replace("\\mathrm", "")
    string = string.replace("\\operatorname", "")
    string = string.replace("\\mathbb", "")
    string = string.replace("\\mathrm", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathcal", "")
    string = string.replace("\\mathsf", "")
    string = string.replace("\\textrm", "")
    string = string.replace("\\mathfrak", "")
    string = string.replace("\\mathnormal", "")
    string = string.replace("\\boldsymbol", "")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\biggl", "")
    string = string.replace("\\biggr", "")
    string = string.replace("\\Biggl", "")
    string = string.replace("\\Biggr", "")
    string = string.replace("\\bigl", "")
    string = string.replace("\\bigr", "")
    string = string.replace("\\Bigl", "")
    string = string.replace("\\Bigr", "")
    string = string.replace("\\tiny", "")
    string = string.replace("\\small", "")
    string = string.replace("\\normalsize", "")
    string = string.replace("\\large", "")
    string = string.replace("\\Large", "")
    string = string.replace("\\LARGE", "")
    string = string.replace("\\Huge", "")
    string = string.replace("\\huge", "")
    string = string.replace("\\displaystyle", "")
    string = string.replace("\\textstyle", "")
    string = string.replace("\\scriptstyle", "")
    string = string.replace("\\scriptscriptstyle", "")
    string = string.replace("\\text", "")
    string = string.replace("\\mbox", "")
    string = string.replace("\\textrm", "")
    string = string.replace("\\textnormal", "")
    string = string.replace("\\mathrm", "")
    string = string.replace("\\mathsf", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\boldsymbol", "")
    string = string.replace("\\mathit", "")
    string = string.replace("\\mathnormal", "")
    string = string.replace("\\mathcal", "")
    string = string.replace("\\mathfrak", "")
    string = string.replace("\\mathbb", "")
    string = string.replace("\\mathtt", "")
    string = string.replace("\\operatorname", "")
    string = string.replace("\\DeclareMathOperator", "")

    string = string.replace("\\frac12", "\\frac{1}{2}")
    string = string.replace("\\frac14", "\\frac{1}{4}")
    string = string.replace("\\frac34", "\\frac{3}{4}")
    string = string.replace("\\frac13", "\\frac{1}{3}")
    string = string.replace("\\frac23", "\\frac{2}{3}")
    string = string.replace("\\frac15", "\\frac{1}{5}")

    string = string.replace("\\dfrac", "\\frac")
    string = string.replace("\\tfrac", "\\frac")
    string = _fix_fracs(string)
    string = _fix_sqrt(string)
    string = _fix_a_slash_b(string)

    string = string.replace("\\ ", "")

    if string and string[0] == "." and (len(string) == 1 or (string[1].isdigit() and "." not in string[1:])):
        string = "0" + string

    return string


def strip_units(simplify_golden: str) -> str:
    for unit_text in unit_texts:
        simplify_golden = simplify_golden.replace(unit_text, "")

    simplify_golden = simplify_golden.replace("sq . in", "")
    simplify_golden = simplify_golden.replace("in . sq", "")
    simplify_golden = simplify_golden.replace("sq . ft", "")

    return simplify_golden


def parse_math_answer(answer: str) -> Dict[str, Any]:
    answer = answer.replace("\n", " ")
    answer = regex.sub(r"\(\s*\\frac", r"\\frac", answer)
    pattern = r"(?:\\boxed|\\fbox)\s*{([^}]*)}"
    matches = regex.findall(pattern, answer)
    if matches:
        result = matches[-1]
    else:
        result = answer
    return {"raw": result, "clean": clean_float_text(strip_units(result))}


def extract_answer(text: str, data_name: str) -> str:
    parse_result = parse_math_answer(text)
    return parse_result["clean"]
