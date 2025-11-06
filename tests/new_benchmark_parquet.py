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
    print("ERROR: pyarrow is required. Please run: pip install pyarrow")
    pq = None


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, "r", encoding="utf-8") as f:
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

                for row_dict in batch.to_pylist():
                    yield row_dict

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
PARQUET_FILE = "reddit_16M.parquet"
DATASET_NAME = "Trimness8/reddit_dataset_145"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"


def setup_file():
    """Downloads and saves the dataset if it doesn't exist."""
    file_path = DATA_DIR / PARQUET_FILE

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
        ds.to_parquet(file_path)
        print(f"File setup complete: {file_path.name}")

    # Get row count for tqdm
    if pq:
        print("Counting rows for progress bar (this is fast for Parquet)...")
        pf = pq.ParquetFile(file_path)
        total_rows = pf.metadata.num_rows
        print(f"Found {total_rows:,} rows.")
        return str(file_path), total_rows
    else:
        return None, 0


def run_benchmark(file_path: str, total_rows: int):
    if file_path is None or total_rows == 0:
        return 0, 0, 0, 0

    file = Path(file_path)
    file_size_mb = file.stat().st_size / (1024 * 1024)
    print(f"\n--- Starting Parquet Benchmark (Optimized): {file.name} ---")
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
    print(f"--- Benchmark Complete ---")
    print(f"  Total Rows: {row_count:,}")
    print(f"  Total Time: {duration:.2f} seconds")
    print(f"  Peak Memory: {peak_memory_mb:.2f} MB")

    return row_count, duration, peak_memory_mb, file_size_mb


def main():
    parquet_path, total_rows = setup_file()
    if not parquet_path:
        print("Failed to set up Parquet file. Exiting.")
        return

    p_rows, p_time, p_mem, p_size = run_benchmark(parquet_path, total_rows)

    results = [
        {
            "Format": "Parquet (Optimized)",
            "Dataset": DATASET_NAME,
            "Rows": f"{p_rows:,}",
            "File Size (MB)": f"{p_size:.2f}",
            "Read Time (sec)": f"{p_time:.2f}",
            "Peak Memory (MB)": f"{p_mem:.2f}",
        }
    ]

    df_results = pd.DataFrame(results)
    print("\n\n--- Parquet Benchmark Summary (Optimized) ---")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
