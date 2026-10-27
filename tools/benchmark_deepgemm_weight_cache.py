"""Measure dense FP8 weight-cache latency/memory; this is not a training benchmark.

Example (requires the GLM alignment DeepGEMM/SGLang environment):
    PYTHONPATH=. python tools/benchmark_deepgemm_weight_cache.py
"""

import argparse
import json
import os
import statistics
import time

import torch

from slime.backends.megatron_utils.alignment.deepgemm_forward import _deepgemm_linear
from slime.backends.megatron_utils.alignment.weight_cache import deepgemm_weight_cache_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("m", 128), ("n", 4096), ("k", 4096), ("microbatches", 8), ("warmup", 3), ("repeats", 7)):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()
    if any(value <= 0 for value in vars(args).values()):
        parser.error("all sizes and iteration counts must be positive")
    os.environ.setdefault("SGLANG_DEEPGEMM_BATCH_INVARIANT", "1")
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    torch.manual_seed(123)
    x = torch.randn((args.m, args.k), device="cuda", dtype=torch.bfloat16) / 16
    weight = torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16) / 16

    @torch.no_grad()
    def run(enabled):
        with deepgemm_weight_cache_scope(enabled):
            for _ in range(args.microbatches):
                output = _deepgemm_linear(x, weight)
        return output

    torch.testing.assert_close(run(False), run(True), rtol=0, atol=0)
    samples = {False: [], True: []}
    peaks = {False: [], True: []}
    # Warm both paths, then alternate their order across measured repetitions.
    for _ in range(args.warmup):
        for enabled in (False, True):
            run(enabled)
    for iteration in range(args.repeats):
        for enabled in (False, True) if iteration % 2 == 0 else (True, False):
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = run(enabled)
            torch.cuda.synchronize()
            samples[enabled].append((time.perf_counter() - start) * 1000)
            peaks[enabled].append(torch.cuda.max_memory_allocated() - baseline)
            del output
    result = {
        "scope": "dense forward only; one fresh cache per microbatch sequence",
        "shape": vars(args),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "output_bitwise_equal": True,
        "measurements": {
            str(enabled).lower(): {
                "median_ms": statistics.median(samples[enabled]),
                "samples_ms": samples[enabled],
                "max_incremental_allocated_bytes": max(peaks[enabled]),
            }
            for enabled in (False, True)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
