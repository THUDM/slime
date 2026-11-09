"""Stable advantage whitening with real CPU collectives."""

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _cp_dist_helpers import free_port

from slime.utils.distributed_utils import distributed_masked_whiten

NUM_GPUS = 0

CASES = {
    "close": ([1.0, 1.0001, 0.9999, 1.0002], [1, 1, 1, 1]),
    "constant": ([0.3] * 6, [1] * 6),
    "offset": ([10000.0, 10000.01, 10000.02, 10000.03], [1] * 4),
    "masked": ([1.0, 10000.0, 1.0001, 0.9999, 1.0002], [1, 0, 1, 1, 1]),
    "ordinary": ([-1.0, 0.0, 1.0, 2.0], [1] * 4),
    "singleton": ([0.3, 10.0], [1, 0]),
}


def _worker(rank, world_size, port, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    results = {}
    try:
        for name, (raw_values, raw_mask) in CASES.items():
            values = torch.tensor(raw_values, dtype=torch.float32)
            mask = torch.tensor(raw_mask, dtype=torch.float32)
            for empty_rank in [False, True]:
                selected = (
                    slice(rank, None, world_size)
                    if not empty_rank or world_size == 1
                    else (slice(0, 0) if rank == 0 else slice(None))
                )
                local_values, local_mask = values[selected], mask[selected]
                for shift_mean in [True, False]:
                    actual = distributed_masked_whiten(local_values, local_mask, shift_mean=shift_mean)
                    assert actual.dtype == local_values.dtype
                    results[f"{name}/{empty_rank}/{shift_mean}"] = actual.tolist()
        try:
            distributed_masked_whiten(torch.ones(1), torch.zeros(1))
        except ValueError:
            results["zero_mask_rejected"] = True
        else:
            results["zero_mask_rejected"] = False
        for dtype_name in ["int64", "float16", "bfloat16", "float64"]:
            values = torch.tensor([-1, 0, 1, 2], dtype=getattr(torch, dtype_name))
            actual = distributed_masked_whiten(values, torch.ones_like(values, dtype=torch.bool))
            results[f"dtype/{dtype_name}"] = {"values": actual.tolist(), "dtype": str(actual.dtype)}
        values = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
        mask = torch.ones_like(values, requires_grad=True)
        gradients = torch.autograd.grad(
            distributed_masked_whiten(values, mask).sum(), (values, mask), allow_unused=True
        )
        results["detached_statistics"] = gradients[1] is None and bool(torch.isfinite(gradients[0]).all())
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=[1, 2])
def distributed_results(request, tmp_path_factory):
    world_size = request.param
    directory = tmp_path_factory.mktemp(f"whiten_{world_size}")
    context = mp.spawn(_worker, args=(world_size, free_port(), str(directory)), nprocs=world_size, join=False)
    deadline = time.monotonic() + 60
    while not context.join(timeout=1):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail("whitening workers did not finish")
    return world_size, [json.loads((directory / f"rank{rank}.json").read_text()) for rank in range(world_size)]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("shift_mean", [True, False])
def test_whiten_matches_centered_reference(distributed_results, name, shift_mean):
    world_size, results = distributed_results
    raw_values, raw_mask = CASES[name]
    values = torch.tensor(raw_values, dtype=torch.float32).double()
    mask = torch.tensor(raw_mask, dtype=torch.bool)
    selected = values[mask]
    mean = selected.mean()
    variance = selected.var(correction=1) if selected.numel() > 1 else selected.new_zeros(())
    expected = (values - mean) / torch.sqrt(variance + 1e-8)
    if not shift_mean:
        expected += mean
    for empty_rank in [False, True]:
        for rank, result in enumerate(results):
            selection = (
                slice(rank, None, world_size)
                if not empty_rank or world_size == 1
                else (slice(0, 0) if rank == 0 else slice(None))
            )
            actual = torch.tensor(result[f"{name}/{empty_rank}/{shift_mean}"])
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected[selection].float(), rtol=2e-5, atol=2e-5)
            assert result["zero_mask_rejected"]


@pytest.mark.parametrize("dtype_name", ["int64", "float16", "bfloat16", "float64"])
def test_whiten_preserves_output_dtype_and_detached_statistics(distributed_results, dtype_name):
    world_size, results = distributed_results
    dtype = getattr(torch, dtype_name)
    output_dtype = torch.float32 if dtype == torch.int64 else dtype
    values = torch.tensor([-1, 0, 1, 2], dtype=torch.float64)
    global_values = values.repeat(world_size)
    expected = ((values - global_values.mean()) / torch.sqrt(global_values.var() + 1e-8)).to(output_dtype)
    tolerance = max(2 * torch.finfo(output_dtype).eps, 2e-5)
    for result in results:
        actual = result[f"dtype/{dtype_name}"]
        assert actual["dtype"] == str(output_dtype)
        torch.testing.assert_close(
            torch.tensor(actual["values"], dtype=output_dtype), expected, rtol=tolerance, atol=tolerance
        )
        assert result["detached_statistics"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
