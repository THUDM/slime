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
    if path.endswith(".parquet"):
        print(f"Reading Parquet with pd.read_parquet: {path}")
        df = pd.read_parquet(path, dtype_backend="pyarrow")
    else:
        raise ValueError(f"This script only supports .parquet files.")

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
PARQUET_FILE = "reddit_16M.parquet"
DATASET_NAME = "Trimness8/reddit_dataset_145"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"


def setup_file():
    file_path = DATA_DIR / PARQUET_FILE
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
    ds.to_parquet(file_path)
    print(f"File setup complete: {file_path.name}")
    return str(file_path)


def run_benchmark(file_path: str):
    if file_path is None:
        return 0, 0, 0, 0

    file = Path(file_path)
    file_size_mb = file.stat().st_size / (1024 * 1024)
    print(f"\n--- Starting Parquet Benchmark: {file.name} ---")
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
    print(f"--- Benchmark Complete ---")
    print(f"  Total Rows: {row_count:,}")
    print(f"  Total Time: {duration:.2f} seconds")
    print(f"  Peak Memory: {peak_memory_mb:.2f} MB")

    return row_count, duration, peak_memory_mb, file_size_mb


def main():
    parquet_path = setup_file()
    if not parquet_path:
        print("Failed to set up Parquet file. Exiting.")
        return

    p_rows, p_time, p_mem, p_size = run_benchmark(parquet_path)

    results = [
        {
            "Format": "Parquet",
            "Dataset": DATASET_NAME,
            "Rows": f"{p_rows:,}",
            "File Size (MB)": f"{p_size:.2f}",
            "Read Time (sec)": f"{p_time:.2f}",
            "Peak Memory (MB)": f"{p_mem:.2f}",
        }
    ]

    df_results = DataFrame(results)
    print("\n\n--- Parquet Benchmark Summary ---")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
