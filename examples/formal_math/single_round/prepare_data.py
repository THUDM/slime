import datetime
import pprint
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import polars as pl
import torch
import typer
from datasets import load_dataset

self_stem = Path(__file__).stem

# https://github.com/deepseek-ai/DeepSeek-Prover-V2
_PROMPT_TEMPLATE = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()


_NEEDLE_THEOREM = "theorem "


def process_flc(
    dir_output: Path,
    train_flc_select_num_rows: int,
    val_flc_select_num_rows: int,
    filter_difficulty: int | None,
    filter_solvable_by_rollout_dumps: str,
):
    ds = load_dataset("m-a-p/FineLeanCorpus", split="train")
    print(f"Loaded dataset: {len(ds)=}")
    ds = _add_metadata_column(ds, dataset_name="flc", column_id="id")

    if (x := filter_solvable_by_rollout_dumps) is not None:
        interesting_question_ids = _SolvableByRolloutDumpFilter.compute_interesting_question_ids(x)

    def _filter_row(lean_code, difficulty, metadata):
        return (
            # we remove multi-theorem data currently
            (lean_code.count(_NEEDLE_THEOREM) == 1)
            and ((filter_difficulty is None) or (difficulty == filter_difficulty))
            and ((not filter_solvable_by_rollout_dumps) or (metadata["question_id"] in interesting_question_ids))
        )

    ds = ds.filter(
        lambda batch: [
            _filter_row(lean_code, difficulty, metadata)
            for lean_code, difficulty, metadata in zip(
                batch["lean_code"], batch["difficulty"], batch["metadata"], strict=True
            )
        ],
        batched=True,
        num_proc=64,
    )
    print(f"Filtered dataset: {len(ds)=}")

    ds = ds.shuffle(seed=42)
    ds = ds.select_columns(["id", "statement", "lean_code", "metadata"])
    ds = ds.select(range(min(len(ds), train_flc_select_num_rows + val_flc_select_num_rows)))
    ds = ds.train_test_split(test_size=val_flc_select_num_rows, shuffle=False, seed=42)
    print(f"Split dataset: {len(ds['train'])=} {len(ds['test'])=}")

    def _process_prompt(statement, lean_code):
        assert lean_code.count(_NEEDLE_THEOREM) == 1, f"{lean_code=}"
        x = lean_code.replace(_NEEDLE_THEOREM, f"/- {statement} -/\n{_NEEDLE_THEOREM}")

        x = _convert_to_by_sorry(x)
        x = _PROMPT_TEMPLATE.format(x)
        x = _to_messages(x)
        return x

    def _process_batch(batch):
        return {
            "prompt": [
                _process_prompt(statement, lean_code)
                for statement, lean_code in zip(batch["statement"], batch["lean_code"], strict=True)
            ]
        }

    ds = ds.map(_process_batch, batched=True, num_proc=64, remove_columns=["statement", "lean_code"])
    _write_file(ds["train"], dir_output / "flc_train.jsonl")
    _write_file(ds["test"], dir_output / "flc_test.jsonl")


class _SolvableByRolloutDumpFilter:
    @staticmethod
    def compute_interesting_question_ids(paths):
        paths = paths.split(",")
        interesting_question_ids = set()
        with ProcessPoolExecutor(max_workers=64) as executor:
            for partial_question_ids in executor.map(
                _SolvableByRolloutDumpFilter._compute_interesting_question_ids_one, paths
            ):
                interesting_question_ids |= partial_question_ids
        print(f"(overall) {len(interesting_question_ids)=} {list(interesting_question_ids)[:5]=}")
        return interesting_question_ids

    @staticmethod
    def _compute_interesting_question_ids_one(paths):
        df_samples = _SolvableByRolloutDumpFilter._compute_df_samples(paths)
        return _SolvableByRolloutDumpFilter._compute_interesting_question_ids_from_df_samples(df_samples)

    @staticmethod
    def _compute_df_samples(paths: str):
        print(f"compute_df_samples {paths=}")
        paths = paths.split(",")
        df_samples = pl.concat([pl.DataFrame(torch.load(p)["samples"]) for p in paths], how="diagonal_relaxed")
        print(f"{df_samples=}")
        return df_samples

    @staticmethod
    def _compute_interesting_question_ids_from_df_samples(df_samples: pl.DataFrame):
        df = df_samples

        df = df.select(
            "prompt",
            "response",
            pl.col("reward").struct.field("reward_value"),
            pl.col("metadata").struct.field("question_id"),
        )
        df = df.group_by("question_id").agg(pl.col("reward_value").mean())

        interesting_question_ids = df.filter(pl.col("reward_value") > 0)["question_id"].sort().to_list()
        print(f"(partial) {len(interesting_question_ids)=} {interesting_question_ids[:5]=}")

        return set(interesting_question_ids)


_LEANABELL_ORIGINAL_PREFIX = """Complete the following Lean 4 code with explanatory comments preceding each line of code:

```lean4
"""


def process_leanabell(
    dir_output: Path,
):
    ds = load_dataset("stoney0062/Leanabell-Prover-Traindata-SFT", split="train")
    ds = _add_metadata_column(ds, dataset_name="leanabell")
    ds = ds.shuffle(seed=42)

    def _compute_messages(raw_prompt, raw_output):
        question_lean = _ensure_remove_prefix(raw_prompt, _LEANABELL_ORIGINAL_PREFIX)
        question_lean = _convert_to_by_sorry(question_lean)

        return [
            {"role": "user", "content": _PROMPT_TEMPLATE.format(question_lean)},
            {"role": "assistant", "content": f"```lean\n{raw_output}"},
        ]

    def _process_batch(batch):
        return {
            "messages": [
                _compute_messages(prompt, output)
                for prompt, output in zip(batch["prompt"], batch["output"], strict=True)
            ]
        }

    ds = ds.map(_process_batch, batched=True, num_proc=64, remove_columns=["prompt", "output"])
    _write_file(ds, dir_output / "leanabell.parquet")


def _ensure_remove_prefix(s: str, prefix: str):
    assert s.startswith(prefix), f"{prefix=} {s=}"
    return s.removeprefix(prefix)


def process_minif2f(
    dir_output: Path,
):
    ds = load_dataset("AI-MO/minif2f_test", split="train")
    ds = _add_metadata_column(ds, dataset_name="minif2f")
    ds = ds.shuffle(seed=42)
    ds = ds.remove_columns(["name", "informal_prefix"])

    def _process_prompt(x):
        x = _convert_to_by_sorry(x)
        x = _PROMPT_TEMPLATE.format(x)
        x = _to_messages(x)
        return x

    def _process_batch(batch):
        return {"prompt": [_process_prompt(x) for x in batch["formal_statement"]]}

    ds = ds.map(_process_batch, batched=True)
    _write_file(ds, dir_output / "minif2f_test.jsonl")


def _write_file(ds, path):
    match path.suffix:
        case ".jsonl":
            ds.to_json(path)
        case ".parquet":
            ds.to_parquet(path)
        case _:
            raise NotImplementedError(f"{path=} {path.suffix=}")

    print(f"Write to {path}, {len(ds)=}, example data:")
    pprint.pprint([ds[i] for i in range(3)])


def _convert_to_by_sorry(s: str):
    return _ensure_remove_pattern(s, r" *:=\s*(?:by\s*)?(?:sorry\s*)?$") + " := by\n  sorry"


def _ensure_remove_pattern(text: str, pattern: str):
    assert re.search(pattern, text, flags=re.MULTILINE), f"{pattern=} {text=}"
    return re.sub(pattern, "", text, flags=re.MULTILINE)


def _to_messages(content):
    return [{"role": "user", "content": content}]


def _add_metadata_column(ds, dataset_name: str, column_id=None):
    if column_id is not None:
        col_metadata = [dict(question_id=f"{dataset_name}__{value}") for value in ds[column_id]]
    else:
        col_metadata = [dict(question_id=f"{dataset_name}__idx{i}") for i in range(len(ds))]
    return ds.add_column("metadata", col_metadata)


def main(
    mode: Annotated[str, typer.Option()] = "rl",
    dir_output_base: Annotated[str, typer.Option()] = "/root/datasets/formal_math_single_round/",
    output_name: Annotated[str, typer.Option()] = None,
    train_flc_select_num_rows: Annotated[int, typer.Option()] = 20000,
    val_flc_select_num_rows: Annotated[int, typer.Option()] = 100,
    filter_difficulty: Annotated[int | None, typer.Option()] = None,
    filter_solvable_by_rollout_dumps: Annotated[str | None, typer.Option()] = None,
):
    dir_output = Path(dir_output_base) / (
        output_name or f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 1000000)}"
    )
    dir_output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {dir_output}")

    if mode == "rl":
        process_flc(
            dir_output=dir_output,
            train_flc_select_num_rows=train_flc_select_num_rows,
            val_flc_select_num_rows=val_flc_select_num_rows,
            filter_difficulty=filter_difficulty,
            filter_solvable_by_rollout_dumps=filter_solvable_by_rollout_dumps,
        )
        process_minif2f(
            dir_output=dir_output,
        )
    elif mode == "sft":
        process_leanabell(
            dir_output=dir_output,
        )
    else:
        raise NotImplementedError(f"{mode=}")


if __name__ == "__main__":
    typer.run(main)
