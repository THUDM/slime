import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_cpu_test_shards(tests, shard_prefix, max_tests=7, split_straw=False):
    shards = []
    regular = [test for test in tests if not test.get("straw", False)]
    for index, start in enumerate(range(0, len(regular), max_tests), start=1):
        shards.append(
            {
                "shard": f"{shard_prefix}-{index}",
                "tests": regular[start : start + max_tests],
                "straw": False,
            }
        )

    straw = [test for test in tests if test.get("straw", False)]
    if split_straw:
        grouped = {}
        ungrouped = []
        for test in straw:
            group = test.get("ci_group")
            if group is None:
                ungrouped.append(test)
            else:
                grouped.setdefault(group, []).append(test)
        for group, group_tests in grouped.items():
            if len(group_tests) > max_tests:
                raise ValueError(f"Straw CI group {group!r} exceeds {max_tests} tests")
            shards.append(
                {
                    "shard": f"{shard_prefix}-straw-{group}",
                    "tests": group_tests,
                    "straw": True,
                }
            )
        for index, start in enumerate(range(0, len(ungrouped), max_tests), start=1):
            shards.append(
                {
                    "shard": f"{shard_prefix}-straw-general-{index}",
                    "tests": ungrouped[start : start + max_tests],
                    "straw": True,
                }
            )
    else:
        for index, start in enumerate(range(0, len(straw), max_tests), start=1):
            shards.append(
                {
                    "shard": f"{shard_prefix}-straw-{index}",
                    "tests": straw[start : start + max_tests],
                    "straw": True,
                }
            )
    return shards


def run_tests(tests, root=ROOT):
    failures = []
    for test in tests:
        if test.get("num_gpus", 0) != 0:
            raise ValueError(f"CPU shard cannot run GPU test {test['test_file']}")
        test_file = test["test_file"]
        test_path = Path(test_file)
        if not test_path.parts or test_path.parts[0] != "tests":
            test_path = Path("tests") / test_path

        command = [sys.executable, str(test_path)]
        test_args = test.get("test_args", "")
        if test_args:
            command.extend(test_args.split())

        print(f"::group::{test_file}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, cwd=root, check=False)
        elapsed = time.perf_counter() - started
        print("::endgroup::", flush=True)
        print(f"{test_file}: exit={result.returncode}, {elapsed:.1f}s", flush=True)
        if result.returncode:
            failures.append((test_file, result.returncode))

    if failures:
        print("Failed tests:", flush=True)
        for test_file, returncode in failures:
            print(f"  {test_file} (exit {returncode})", flush=True)
    else:
        print(f"All {len(tests)} tests passed.", flush=True)
    return int(bool(failures))


def main():
    shard = json.loads(os.environ["SLIME_CPU_TESTS"])
    if not isinstance(shard, dict) or not shard.get("shard"):
        raise ValueError("SLIME_CPU_TESTS must contain a named shard object")
    tests = shard.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError("CPU shard must contain a non-empty test list")
    print(f"Running {shard['shard']} ({len(tests)} tests)", flush=True)
    return run_tests(tests)


if __name__ == "__main__":
    raise SystemExit(main())
