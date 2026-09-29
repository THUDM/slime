"""Add the calculator agent's system prompt to GSM8K.

    python prepare_data.py SRC_DIR OUT_DIR

SRC_DIR holds zhuzilin/gsm8k (``train.parquet`` and ``test.parquet``, a ``messages`` chat list and a
``label``); OUT_DIR receives the same files with the agent's system prompt in place of the original one.
"""

import sys
from pathlib import Path

import pandas as pd
from calculator_agent import SYSTEM_PROMPT


def main():
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        frame = pd.read_parquet(source / f"{split}.parquet")
        frame["messages"] = [
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + [dict(message) for message in messages if message["role"] != "system"]
            for messages in frame["messages"]
        ]
        frame.to_parquet(target / f"{split}.parquet")
        print(f"{split}: {len(frame)} rows -> {target / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
