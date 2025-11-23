# Adapt from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/verl/utils/reward_score/qa_em_format.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import random
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)

    if not assistant_match:
        return False, "Missing assistant marker"

    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]

    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end

    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are no matches, return None
    if len(matches) == 0:
        return None

    # Return the last answer (handles both single and multiple answers)
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(
    solution_str,
    ground_truth,
    method="strict",
    structure_format_score=0,
    final_format_score=0,
    retrieval_score=0,
    format_score=0,
    score=1.0,
    multi_answer_penalty_per_extra=0.05,
    multi_answer_penalty_cap=0.2,
):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        multi_answer_penalty_per_extra: penalty per extra <answer> tag beyond the first
        multi_answer_penalty_cap: maximum penalty for multiple <answer> tags
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])
    answer = extract_solution(solution_str=solution_str)

    # Count number of <answer> tags for progressive penalty
    num_answers = len(re.findall(r"<answer>", solution_str))
    multi_answer_penalty = min(
        multi_answer_penalty_cap,
        multi_answer_penalty_per_extra * max(0, num_answers - 1)
    )

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Num answers: {num_answers}, penalty: {multi_answer_penalty}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score  # 0.3
            else:
                return structure_format_score  # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if is_valid_format:
                # Apply progressive penalty for multiple <answer> tags
                return score - multi_answer_penalty
            else:
                return score - structure_format_score - multi_answer_penalty
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score - multi_answer_penalty
            else:
                return structure_format_score - multi_answer_penalty
        else:
            return max(0, final_format_score - multi_answer_penalty)


def log_turn_by_turn(sample, show_full_content: bool = False):
    """Print turn-by-turn rollout log based on OpenAI messages format.

    Args:
        sample: Sample object with metadata["messages"] containing OpenAI format messages
        show_full_content: If True, show full content; otherwise truncate for readability
    """
    messages = sample.metadata.get("messages", [])

    if not messages:
        print("⚠️  No messages found in sample.metadata")
        return

    print("\n" + "="*70)
    print("Turn-by-Turn Rollout Analysis")
    print("="*70)

    turn = 0
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")

        if role == "user":
            print(f"\n[User Prompt]")
            content = msg.get("content", "")
            if show_full_content:
                print(f"  {content}")
            else:
                print(f"  {content[:150]}..." if len(content) > 150 else f"  {content}")

        elif role == "assistant":
            turn += 1
            print(f"\n--- Turn {turn}: Assistant ---")

            content = msg.get("content")
            if content:
                if show_full_content:
                    print(f"Content: {content}")
                else:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"Content: {preview}")

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                print(f"Tool Calls: {len(tool_calls)} call(s)")
                for idx, tc in enumerate(tool_calls):
                    func_name = tc.get("function", {}).get("name", "unknown")
                    args = tc.get("function", {}).get("arguments", "")
                    print(f"  [{idx+1}] Function: {func_name}")
                    if show_full_content:
                        print(f"      Arguments: {args}")
                    else:
                        args_preview = args[:100] + "..." if len(args) > 100 else args
                        print(f"      Arguments: {args_preview}")

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "N/A")
            content = msg.get("content", "")
            print(f"\n  → Tool Result (id={tool_call_id}):")
            if show_full_content:
                print(f"    {content}")
            else:
                preview = content[:300] + "..." if len(content) > 300 else content
                print(f"    {preview}")

    # Summary
    print("\n" + "-"*70)
    print("Summary")
    print("-"*70)
    print(f"Total turns: {turn}")
    print(f"Total tool calls: {sample.metadata.get('total_tool_calls', 0)}")
    print(f"Has final <answer>: {sample.metadata.get('has_final_answer', False)}")
    print(f"Final turn has tool_calls (ERROR): {sample.metadata.get('final_turn_has_tool_calls', False)}")
    print(f"Response status: {sample.status.value if hasattr(sample.status, 'value') else sample.status}")
    print("="*70 + "\n")
