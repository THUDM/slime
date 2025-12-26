import itertools
import json
import re
import time
import tracemalloc
import warnings
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        print(f"read_file path={path} applying slice {row_slice=}")
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


DATA_DIR = Path("./benchmark_data")
JSONL_FILE = "fineweb_instruct.jsonl"
DATASET_NAME = "TIGER-Lab/Fineweb-Instruct"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"


def setup_file():
    """Downloads and saves the dataset if it doesn't exist."""
    file_path = DATA_DIR / JSONL_FILE

    if file_path.exists():
        print(f"Benchmark file '{file_path.name}' already exists. Skipping creation.")
    else:
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

    print("Counting rows for progress bar (this may take a minute)...")
    total_rows = 0
    with open(file_path, encoding="utf-8") as f:
        for _line in f:
            total_rows += 1
    print(f"Found {total_rows:,} rows.")
    return str(file_path), total_rows


def run_benchmark(file_path: str, total_rows: int):
    if file_path is None or total_rows == 0:
        return 0, 0, 0, 0

    file = Path(file_path)
    file_size_mb = file.stat().st_size / (1024 * 1024)
    print(f"\n--- Starting JSONL Benchmark (Optimized): {file.name} ---")
    print(f"  File Size: {file_size_mb:.2f} MB")

    tracemalloc.start()
    start_time = time.perf_counter()

    row_count = 0
    try:

        for _ in tqdm(read_file(file_path), total=total_rows, desc="Streaming rows"):
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
    jsonl_path, total_rows = setup_file()
    if not jsonl_path:
        print("Failed to set up JSONL file. Exiting.")
        return

    j_rows, j_time, j_mem, j_size = run_benchmark(jsonl_path, total_rows)

    results = [
        {
            "Format": "JSONL (Optimized)",
            "Dataset": f"{DATASET_NAME}",
            "Rows": f"{j_rows:,}",
            "File Size (MB)": f"{j_size:.2f}",
            "Read Time (sec)": f"{j_time:.2f}",
            "Peak Memory (MB)": f"{j_mem:.2f}",
        }
    ]

    df_results = pd.DataFrame(results)
    print("\n\n--- JSONL Benchmark Summary (Optimized) ---")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
