from __future__ import annotations

import ast
import re



def _normalize_instruction_ids(raw_ids):
    out = []
    for value in raw_ids or []:
        text = str(value).strip() if value is not None else ""
        if text:
            out.append(text)
    return out


def _normalize_kwargs(raw_kwargs, n):
    if isinstance(raw_kwargs, list):
        items = [dict(x) if isinstance(x, dict) else {} for x in raw_kwargs]
    elif isinstance(raw_kwargs, dict):
        items = [dict(raw_kwargs) for _ in range(n)]
    else:
        items = [{} for _ in range(n)]

    if len(items) < n:
        items.extend({} for _ in range(n - len(items)))
    elif len(items) > n:
        items = items[:n]

    return [{k: v for k, v in item.items() if v is not None} for item in items]


def _clean_response(text: str) -> str:
    text = text or ""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx >= 0:
        text = text[idx + len(marker) :]
    return text.strip().replace("<|endoftext|>", "").replace("<pad>", "").replace("<|im_end|>", "").strip()


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text, flags=re.UNICODE)


def _normalized_words(text: str) -> list[str]:
    return [w.lower() for w in _word_tokens(text)]


def _count_keyword(text: str, keyword: str) -> int:
    return sum(1 for w in _normalized_words(text) if w == str(keyword).lower())


def _relation_ok(actual: int, target: int, relation: str | None) -> bool:
    relation = (relation or "exactly").strip().lower()
    if relation in {"exactly", "equal", "equal to"}:
        return actual == target
    if relation in {"at least", "greater than or equal", "no less than"}:
        return actual >= target
    if relation in {"at most", "less than or equal", "no more than"}:
        return actual <= target
    if relation in {"less than", "fewer than"}:
        return actual < target
    if relation in {"more than", "greater than"}:
        return actual > target
    if relation == "around":
        tolerance = max(round(target * 0.1), 1)
        return abs(actual - target) <= tolerance
    return actual == target


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_paragraphs(text: str) -> list[str]:
    for divider in ("* * *", "***"):
        if divider in text:
            return [part.strip() for part in text.split(divider) if part.strip()]
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def _first_word(text: str) -> str:
    words = _word_tokens(text)
    return words[0].lower() if words else ""


def _last_word(text: str) -> str:
    words = _word_tokens(text)
    return words[-1].lower() if words else ""


def _extract_json_payload(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _expected_paragraphs_from_prompt(prompt: str) -> int | None:
    match = re.search(r"There should be (\d+) paragraphs", prompt, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _verify_keywords(text: str, keyword_list: list[str]) -> bool:
    lower = text.lower()
    return all(str(keyword).lower() in lower for keyword in keyword_list)


def _validate_forbidden_words(text: str, forbidden_words: list[str]) -> bool:
    lower = text.lower()
    return all(str(word).lower() not in lower for word in forbidden_words)


def _verify_letter_frequency(text: str, letter: str, frequency: int, relation: str) -> bool:
    actual = text.lower().count(str(letter).lower())
    return _relation_ok(actual, int(frequency), relation)


def _verify_total_letter_count(text: str, n: int, relation: str) -> bool:
    actual = sum(1 for ch in text if ch.isalpha())
    return _relation_ok(actual, int(n), relation)


def _verify_lowercase_word_count(text: str, n: int) -> bool:
    actual = len(re.findall(r"\b[a-z]+\b", text))
    return actual <= int(n)


def _verify_paragraph_count(text: str, n: int) -> bool:
    return len(_split_paragraphs(text)) == int(n)


def _validate_word_constraint(text: str, n: int, relation: str) -> bool:
    return _relation_ok(len(_word_tokens(text)), int(n), relation)


def _verify_sentence_constraint(text: str, n: int, relation: str) -> bool:
    return _relation_ok(len(_split_sentences(text)), int(n), relation)


def _validate_paragraphs(text: str, num_paragraphs: int, first_word: str, nth_paragraph: int) -> bool:
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) != int(num_paragraphs):
        return False
    idx = int(nth_paragraph) - 1
    if idx < 0 or idx >= len(paragraphs):
        return False
    return _first_word(paragraphs[idx]) == str(first_word).lower()


def _verify_postscript(text: str, postscript_marker: str) -> bool:
    marker = str(postscript_marker)
    idx = text.find(marker)
    if idx < 0:
        return False
    return len(text[idx:].strip()) > len(marker)


def _validate_placeholders(text: str, n: int) -> bool:
    return len(re.findall(r"\[[^\[\]]+\]", text)) >= int(n)


def _verify_bullet_points(text: str, n: int) -> bool:
    bullets = [line for line in text.splitlines() if line.strip().startswith(("*", "-"))]
    return len(bullets) == int(n)


def _validate_title(text: str) -> bool:
    return re.search(r"<<[^<>]+>>", text) is not None


def _validate_choice(text: str, prompt: str) -> bool:
    match = re.search(r"one of the following options:\s*(\(.+?\))", prompt, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return False
    try:
        options = ast.literal_eval(match.group(1))
    except Exception:
        return False
    stripped = text.strip()
    return stripped in {str(option).strip() for option in options}


def _validate_highlighted_sections(text: str, n: int) -> bool:
    return len(re.findall(r"\*[^*\n]+\*", text)) >= int(n)


def _validate_sections(text: str, n: int, section_splitter: str) -> bool:
    splitter = str(section_splitter)
    matches = re.findall(rf"(?m)^\s*{re.escape(splitter)}\s+\d+\b", text)
    return len(matches) == int(n)


def _validate_json_format(text: str) -> bool:
    import json

    try:
        json.loads(_extract_json_payload(text))
        return True
    except Exception:
        return False


def _validate_repeat_prompt(text: str, prompt: str) -> bool:
    return text.strip().startswith(prompt.strip())


def _validate_two_responses(text: str) -> bool:
    if text.count("******") != 1:
        return False
    first, second = [part.strip() for part in text.split("******", 1)]
    return bool(first) and bool(second) and first != second


def _validate_uppercase(text: str) -> bool:
    return text == text.upper()


def _validate_lowercase(text: str) -> bool:
    return text == text.lower()


def _validate_frequency_capital_words(text: str, n: int, relation: str) -> bool:
    actual = len(re.findall(r"\b[A-Z]+\b", text))
    return _relation_ok(actual, int(n), relation)


def _validate_end(text: str, end_phrase: str) -> bool:
    return text.rstrip().endswith(str(end_phrase))


def _validate_quotation(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith('"') and stripped.endswith('"')


def _validate_no_commas(text: str) -> bool:
    return "," not in text


def _validate_no_char(text: str, forbidden: str) -> bool:
    return forbidden not in text


def _validate_same_start_end_word(text: str) -> bool:
    words = _normalized_words(text)
    return len(words) >= 2 and words[0] == words[-1]


def _validate_first_word_answer(text: str, first_word: str) -> bool:
    return _first_word(text) == str(first_word).lower()


def _validate_last_word_answer(text: str, last_word: str) -> bool:
    return _last_word(text) == str(last_word).lower()


def _validate_first_word_each_sentence(text: str, first_word: str) -> bool:
    expected = str(first_word).lower()
    sentences = _split_sentences(text)
    return bool(sentences) and all(_first_word(sentence) == expected for sentence in sentences)


def _validate_last_word_each_sentence(text: str, last_word: str) -> bool:
    expected = str(last_word).lower()
    sentences = _split_sentences(text)
    return bool(sentences) and all(_last_word(sentence) == expected for sentence in sentences)


def _validate_unique_words(text: str) -> bool:
    words = _normalized_words(text)
    return len(words) == len(set(words))


def _validate_square_brackets(text: str) -> bool:
    chunks = re.findall(r"\[[^\[\]\s]+\]", text)
    compact = re.sub(r"\s+", "", text)
    return bool(chunks) and "".join(chunks) == compact


def _validate_bigram_wrapping(text: str) -> bool:
    matches = re.findall(r"<<([^<>]+)>>", text)
    if not matches:
        return False
    compact = re.sub(r"\s+", "", text)
    rebuilt = "".join(f"<<{match}>>" for match in matches)
    if rebuilt != compact:
        return False
    return all(len(_word_tokens(match)) == 2 for match in matches)


def _validate_sentence_hyphens(text: str) -> bool:
    return "-" in text and " - " not in text and "- " not in text and " -" not in text


def _validate_palindrome(text: str) -> bool:
    for word in _normalized_words(text):
        if len(word) >= 3 and word == word[::-1]:
            return True
    return False


def _validate_no_adjacent_consecutive(text: str) -> bool:
    words = _normalized_words(text)
    initials = [word[0] for word in words if word and word[0].isalpha()]
    for left, right in zip(initials, initials[1:]):
        if abs(ord(left) - ord(right)) == 1:
            return False
    return True


def _validate_keyword_specific_position(text: str, keyword: str, n: int, m: int) -> bool:
    sentences = _split_sentences(text)
    idx = int(n) - 1
    if idx < 0 or idx >= len(sentences):
        return False
    words = _normalized_words(sentences[idx])
    pos = int(m) - 1
    if pos < 0 or pos >= len(words):
        return False
    return words[pos] == str(keyword).lower()


def _validate_repeat_phrase(text: str, phrase: str, small_n: int) -> bool:
    base = _normalized_words(phrase)
    if not base:
        return False
    center = len(base) // 2
    words = _normalized_words(text)
    matches = 0
    for i in range(0, len(words) - len(base) + 1):
        window = words[i : i + len(base)]
        if all(window[j] == base[j] for j in range(len(base)) if j != center) and window[center] != base[center]:
            matches += 1
    return matches == int(small_n)


def _validate_counting_composition(text: str, prompt: str, n_sent: int, n_words: int) -> bool:
    expected_paragraphs = _expected_paragraphs_from_prompt(prompt)
    paragraphs = _split_paragraphs(text)
    if expected_paragraphs is not None and len(paragraphs) != expected_paragraphs:
        return False
    for paragraph in paragraphs:
        sentences = _split_sentences(paragraph)
        if len(sentences) != int(n_sent):
            return False
        for sentence in sentences:
            if len(_word_tokens(sentence)) != int(n_words):
                return False
    return bool(paragraphs)


def _validate_response_language(text: str, language: str) -> bool:
    from langdetect import detect

    try:
        return detect(text) == language
    except Exception:
        return False


def _check_instruction(instruction_id: str, rule_kwargs: dict, prompt: str, response: str) -> bool:
    if instruction_id == "keywords:existence":
        return _verify_keywords(response, rule_kwargs.get("keywords") or [])
    if instruction_id == "keywords:forbidden_words":
        return _validate_forbidden_words(response, rule_kwargs.get("forbidden_words") or [])
    if instruction_id == "punctuation:no_comma":
        return _validate_no_commas(response)
    if instruction_id == "detectable_format:title":
        return _validate_title(response)
    if instruction_id == "detectable_format:bigram_wrapping":
        return _validate_bigram_wrapping(response)
    if instruction_id == "keywords:letter_frequency":
        return _verify_letter_frequency(
            response,
            rule_kwargs.get("letter", ""),
            int(rule_kwargs.get("let_frequency", 0)),
            rule_kwargs.get("let_relation", "exactly"),
        )
    if instruction_id == "punctuation:punctuation_exclamation":
        return _validate_no_char(response, "!")
    if instruction_id == "last_word:last_word_answer":
        return _validate_last_word_answer(response, rule_kwargs.get("last_word", ""))
    if instruction_id == "count:lowercase_counting":
        return _verify_lowercase_word_count(response, int(rule_kwargs.get("N", 0)))
    if instruction_id in {"keywords:word_count_different_numbers", "keywords:frequency"}:
        return _relation_ok(
            _count_keyword(response, rule_kwargs.get("keyword", "")),
            int(rule_kwargs.get("frequency", 0)),
            rule_kwargs.get("relation", "exactly"),
        )
    if instruction_id == "startend:end_checker":
        return _validate_end(response, rule_kwargs.get("end_phrase", ""))
    if instruction_id == "punctuation:punctuation_dot":
        return _validate_no_char(response, ".")
    if instruction_id == "keywords:word_once":
        return _count_keyword(response, rule_kwargs.get("keyword", "")) == 1
    if instruction_id == "detectable_format:sentence_hyphens":
        return _validate_sentence_hyphens(response)
    if instruction_id == "keywords:palindrome":
        return _validate_palindrome(response)
    if instruction_id == "detectable_format:number_bullet_lists":
        return _verify_bullet_points(response, int(rule_kwargs.get("num_bullets", 0)))
    if instruction_id == "letters:letter_counting2":
        return _verify_letter_frequency(
            response,
            rule_kwargs.get("letter", ""),
            int(rule_kwargs.get("let_frequency", 0)),
            rule_kwargs.get("let_relation", "exactly"),
        )
    if instruction_id == "keywords:keyword_specific_position":
        return _validate_keyword_specific_position(
            response,
            rule_kwargs.get("keyword", ""),
            int(rule_kwargs.get("n", 0)),
            int(rule_kwargs.get("m", 0)),
        )
    if instruction_id == "length_constraints:number_words":
        return _validate_word_constraint(
            response,
            int(rule_kwargs.get("num_words", 0)),
            rule_kwargs.get("relation", "exactly"),
        )
    if instruction_id == "keywords:start_end":
        return _validate_same_start_end_word(response)
    if instruction_id == "detectable_format:square_brackets":
        return _validate_square_brackets(response)
    if instruction_id == "detectable_format:number_highlighted_sections":
        return _validate_highlighted_sections(response, int(rule_kwargs.get("num_highlights", 0)))
    if instruction_id == "detectable_content:number_placeholders":
        return _validate_placeholders(response, int(rule_kwargs.get("num_placeholders", 0)))
    if instruction_id == "copy:repeat_phrase":
        return _validate_repeat_phrase(
            response,
            rule_kwargs.get("phrase", ""),
            int(rule_kwargs.get("small_n", 0)),
        )
    if instruction_id == "startend:quotation":
        return _validate_quotation(response)
    if instruction_id == "first_word:first_word_answer":
        return _validate_first_word_answer(response, rule_kwargs.get("first_word", ""))
    if instruction_id == "detectable_content:postscript":
        return _verify_postscript(response, rule_kwargs.get("postscript_marker", ""))
    if instruction_id == "language:response_language":
        return _validate_response_language(response, rule_kwargs.get("language", ""))
    if instruction_id == "keywords:no_adjacent_consecutive":
        return _validate_no_adjacent_consecutive(response)
    if instruction_id == "count:count_increment_word":
        return (
            _count_keyword(response, rule_kwargs.get("keyword1", "")) == 1
            and _count_keyword(response, rule_kwargs.get("keyword2", "")) == 2
        )
    if instruction_id == "letters:letter_counting":
        return _verify_total_letter_count(
            response,
            int(rule_kwargs.get("N", 0)),
            rule_kwargs.get("relation", "exactly"),
        )
    if instruction_id == "length_constraints:number_paragraphs":
        return _verify_paragraph_count(response, int(rule_kwargs.get("num_paragraphs", 0)))
    if instruction_id == "length_constraints:number_sentences":
        return _verify_sentence_constraint(
            response,
            int(rule_kwargs.get("num_sentences", 0)),
            rule_kwargs.get("relation", "exactly"),
        )
    if instruction_id == "change_case:english_capital":
        return _validate_uppercase(response)
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        return _validate_paragraphs(
            response,
            int(rule_kwargs.get("num_paragraphs", 0)),
            rule_kwargs.get("first_word", ""),
            int(rule_kwargs.get("nth_paragraph", 0)),
        )
    if instruction_id == "change_case:english_lowercase":
        return _validate_lowercase(response)
    if instruction_id == "first_word:first_word_sent":
        return _validate_first_word_each_sentence(response, rule_kwargs.get("first_word", ""))
    if instruction_id == "last_word:last_word_sent":
        return _validate_last_word_each_sentence(response, rule_kwargs.get("last_word", ""))
    if instruction_id == "detectable_format:multiple_sections":
        return _validate_sections(
            response,
            int(rule_kwargs.get("num_sections", 0)),
            rule_kwargs.get("section_spliter", rule_kwargs.get("section_splitter", "")),
        )
    if instruction_id == "count:count_unique":
        return _validate_unique_words(response)
    if instruction_id in {"paragraphs:paragraphs", "paragraphs:paragraphs2"}:
        expected = _expected_paragraphs_from_prompt(prompt)
        return _verify_paragraph_count(response, expected) if expected is not None else False
    if instruction_id == "change_case:capital_word_frequency":
        return _validate_frequency_capital_words(
            response,
            int(rule_kwargs.get("capital_frequency", 0)),
            rule_kwargs.get("capital_relation", "exactly"),
        )
    if instruction_id == "count:counting_composition":
        return _validate_counting_composition(
            response,
            prompt,
            int(rule_kwargs.get("n_sent", 0)),
            int(rule_kwargs.get("n_words", 0)),
        )
    if instruction_id == "combination:two_responses":
        return _validate_two_responses(response)
    if instruction_id == "detectable_format:json_format":
        return _validate_json_format(response)
    if instruction_id == "detectable_format:constrained_response":
        return _validate_choice(response, prompt)
    return False


async def reward_func(args, sample, **kwargs):
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        return 0.0

    prompt_text = metadata.get("prompt_text") or sample.prompt or ""
    prompt_text = prompt_text if isinstance(prompt_text, str) else str(prompt_text)
    kwargs_list = _normalize_kwargs(metadata.get("kwargs"), len(instruction_ids))
    response = _clean_response(sample.response or "")
    if not response:
        return 0.0

    passed = []
    for instruction_id, rule_kwargs in zip(instruction_ids, kwargs_list):
        try:
            passed.append(1.0 if _check_instruction(instruction_id, rule_kwargs, prompt_text, response) else 0.0)
        except Exception:
            passed.append(0.0)

    return sum(passed) / len(passed)
