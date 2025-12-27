import re
import time
import tracemalloc
import warnings

# Mocking Sample type for standalone execution
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from pandas import DataFrame
from tqdm import tqdm


@dataclass
class Sample:
    prompt: str
    label: str = None
    metadata: dict = None


# TODO: don't read the whole file into memory.
def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    if path.endswith(".jsonl"):
        print(f"Reading JSONL with pd.read_json: {path}")
        df = pd.read_json(path, lines=True, dtype={"label": str})
    else:
        raise ValueError("This script only supports .jsonl files.")

    if row_slice is not None:
        df = df.iloc[row_slice]

    print("Loading complete. Starting iteration...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Iterating rows"):
        yield row.to_dict()


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)
    return s, None


DATA_DIR = Path("./benchmark_data")
JSONL_FILE = "fineweb_instruct_10M.jsonl"
DATASET_NAME = "TIGER-Lab/Fineweb-Instruct"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"


def setup_file():
    file_path = DATA_DIR / JSONL_FILE
    if file_path.exists():
        print(f"Benchmark file '{file_path.name}' already exists. Skipping creation.")
        return str(file_path)

    print(f"\nSetting up file: {file_path.name}")
    DATA_DIR.mkdir(exist_ok=True)
    warnings.simplefilter("ignore")

    print(f"Loading '{DATASET_NAME}' (config={DATASET_CONFIG}, split={DATASET_SPLIT})...")
    try:
        ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load '{DATASET_NAME}' dataset: {e}")
        return None

    print(f"Dataset loaded with {len(ds):,} rows.")
    print(f"Saving {len(ds):,} rows to {file_path}...")

    ds.to_json(file_path, lines=True)

    print(f"File setup complete: {file_path.name}")
    return str(file_path)


def run_benchmark(file_path: str):
    if file_path is None:
        return 0, 0, 0, 0

    file = Path(file_path)
    file_size_mb = file.stat().st_size / (1024 * 1024)
    print(f"\n--- Starting JSONL Benchmark: {file.name} ---")
    print(f"  File Size: {file_size_mb:.2f} MB")

    tracemalloc.start()
    start_time = time.perf_counter()

    row_count = 0
    try:
        for _ in read_file(file_path):
            row_count += 1
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        tracemalloc.stop()
        return 0, 0, 0, 0

    duration = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak / (1024 * 1024)
    print("--- Benchmark Complete ---")
    print(f"  Total Rows: {row_count:,}")
    print(f"  Total Time: {duration:.2f} seconds")
    print(f"  Peak Memory: {peak_memory_mb:.2f} MB")

    return row_count, duration, peak_memory_mb, file_size_mb


def main():
    jsonl_path = setup_file()
    if not jsonl_path:
        print("Failed to set up JSONL file. Exiting.")
        return

    j_rows, j_time, j_mem, j_size = run_benchmark(jsonl_path)

    results = [
        {
            "Format": "JSONL",
            "Dataset": f"{DATASET_NAME}",
            "Rows": f"{j_rows:,}",
            "File Size (MB)": f"{j_size:.2f}",
            "Read Time (sec)": f"{j_time:.2f}",
            "Peak Memory (MB)": f"{j_mem:.2f}",
        }
    ]

    df_results = DataFrame(results)
    print("\n\n--- JSONL Benchmark Summary ---")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
