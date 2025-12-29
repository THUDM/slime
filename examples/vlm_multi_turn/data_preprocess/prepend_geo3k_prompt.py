#!/usr/bin/env python3
"""Prepend a fixed instruction block to problem fields in Geo3k parquets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROMPT = (
    "You are a math/geometry expert. Solve the user's question carefully and verify your work.\n\n"
    "Follow this protocol:\n"
    "1) First, reason step by step as an internal monologue wrapped inside <think>...</think> tags.\n"
    "2) After you finish the reasoning, you must call the `calc_score` (aka `calc_geo3k_reward`) tool at least once with your parsed numeric answer to check correctness before finalizing.\n"
    "   - Emit tool calls in the below format:\n"
    "    <tool_call>{\"name\": \"calc_score\", \"arguments\": {\"answer\": \"<digits>\"}}</tool_call>\n"
    "    Always include the <tool_call></tool_call> tag, and object within the tag must be JSON.\n"
    "    Right after emitting tool call (in the same message), you must always also include the answer in the form Answer: \\\boxed{$Answer} so the answer can be extracted.\n"
    "3) Use the tool feedback to refine your solution if needed.\n"
    "4) Provide the final answer in the form Answer: \\\boxed{$Answer} where $Answer is the answer to the problem."
)


def _prepend_prompt(df: pd.DataFrame, prompt: str) -> pd.DataFrame:
    if "problem" not in df.columns:
        raise KeyError("Missing 'problem' column in parquet")
    df = df.copy()
    df["problem"] = prompt + "\n\n" + df["problem"].astype(str)
    return df


def main() -> None:
    train_path = Path("/root/datasets/geo3k_imgurl/train.parquet")
    test_path = Path("/root/datasets/geo3k_imgurl/test.parquet")

    train_df = pd.read_parquet(train_path)
    if train_df.empty:
        raise ValueError("train.parquet is empty")

    first_problem = str(train_df.iloc[0]["problem"])
    if PROMPT in first_problem:
        print("Failed tot prepend the tool call prompt: the tool call prompt already prepended.")
        return

    train_df = _prepend_prompt(train_df, PROMPT)
    test_df = _prepend_prompt(pd.read_parquet(test_path), PROMPT)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("Prepended prompt to train/test parquets.")


if __name__ == "__main__":
    main()
