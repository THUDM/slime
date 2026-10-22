from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.backends.megatron_utils.update_weight.weight_sync_credit import WeightSyncCreditController

NUM_GPUS = 0


def _nccl_lifecycle_worker(rank: int, world_size: int, init_method: str, artifact_dir: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
    control_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    bucket_elements = (257, 1024)
    element_size = torch.empty((), dtype=torch.float32).element_size()
    controller = WeightSyncCreditController(
        max_inflight_buckets=2,
        max_inflight_bytes=sum(bucket_elements) * element_size,
    )
    try:
        device = torch.device("cuda", rank)
        torch.cuda.reset_peak_memory_stats(device)
        sync_started = time.perf_counter()
        tensors = [
            (
                torch.full((elements,), float(bucket_id + 1), device=device)
                if rank == 0
                else torch.zeros(elements, device=device)
            )
            for bucket_id, elements in enumerate(bucket_elements)
        ]
        works = []
        reservations = []
        if rank == 0:
            controller.begin_version(17)

        for tensor in tensors:
            bucket_bytes = tensor.numel() * tensor.element_size()
            if rank == 0:
                reservation = controller.reserve(bucket_bytes)
                assert reservation is not None
                controller.mark_launched(
                    reservation,
                    transport_bytes=bucket_bytes,
                    staging_bytes=bucket_bytes,
                    consumer_objects=world_size - 1,
                )
                reservations.append(reservation)
            works.append(dist.broadcast(tensor, src=0, async_op=True))

        if rank == 0:
            launched = controller.snapshot
            assert launched.inflight_buckets == 2
            assert launched.transport_outstanding_bytes == sum(bucket_elements) * element_size
            assert launched.pending_consumer_objects == world_size - 1 + world_size - 1

        for bucket_id, (tensor, work) in enumerate(zip(tensors, works, strict=True)):
            work.wait()
            if rank == 0:
                reservation = reservations[bucket_id]
                controller.mark_transport_complete(reservation)
                # The consumer cannot acknowledge until the source has observed
                # that transport is still distinct from consumer completion.
                assert controller.snapshot.pending_consumer_objects > 0
                ready = [bucket_id]
                dist.broadcast_object_list(ready, src=0, group=control_group)
                acknowledgements = [None] * world_size
                dist.all_gather_object(acknowledgements, f"rank-{rank}-loaded-{bucket_id}", group=control_group)
                assert acknowledgements == [f"rank-{member}-loaded-{bucket_id}" for member in range(world_size)]
                controller.mark_consumers_complete(reservation)
                controller.mark_staging_released(reservation)
                controller.release(reservation)
            else:
                ready = [None]
                dist.broadcast_object_list(ready, src=0, group=control_group)
                assert ready[0] == bucket_id
                torch.testing.assert_close(tensor, torch.full_like(tensor, float(bucket_id + 1)))
                acknowledgements = [None] * world_size
                dist.all_gather_object(acknowledgements, f"rank-{rank}-loaded-{bucket_id}", group=control_group)

        if rank == 0:
            controller.commit_version(17)
            snapshot = controller.snapshot
            assert snapshot.active_version is None
            assert snapshot.inflight_buckets == 0
            assert snapshot.transport_outstanding_bytes == 0
            assert snapshot.staging_resident_bytes == 0
            assert snapshot.pending_consumer_objects == 0
            metrics = controller.metrics()
        else:
            metrics = {}

        torch.cuda.synchronize(device)
        sync_seconds = time.perf_counter() - sync_started

        report = {
            "backend": dist.get_backend(),
            "bucket_bytes": [elements * element_size for elements in bucket_elements],
            "committed_version": 17 if rank == 0 else None,
            "control_backend": dist.get_backend(control_group),
            "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "metrics": metrics,
            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "physical_gpu_uuids": os.environ.get("PHYSICAL_GPU_UUIDS", "unknown"),
            "process_group_membership": list(range(world_size)),
            "rank": rank,
            "sync_seconds": sync_seconds,
            "torch_version": torch.__version__,
            "world_size": world_size,
        }
        output = Path(artifact_dir) / f"weight-sync-lifecycle-nccl-rank-{rank}.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("SLIME_RUN_WEIGHT_SYNC_LIFECYCLE_NCCL_TEST") != "1",
    reason="set SLIME_RUN_WEIGHT_SYNC_LIFECYCLE_NCCL_TEST=1 for the opt-in NCCL check",
)
def test_weight_sync_lifecycle_with_nccl_transport(tmp_path: Path) -> None:
    world_size = int(os.environ.get("SLIME_WEIGHT_SYNC_LIFECYCLE_WORLD_SIZE", "2"))
    if world_size not in (2, 4):
        pytest.fail("SLIME_WEIGHT_SYNC_LIFECYCLE_WORLD_SIZE must be 2 or 4")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"the NCCL check requires {world_size} visible CUDA devices")
    artifact_dir = os.environ.get("SLIME_WEIGHT_SYNC_LIFECYCLE_ARTIFACT_DIR", str(tmp_path))
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _nccl_lifecycle_worker,
        args=(world_size, f"file://{tmp_path / 'nccl_init'}", artifact_dir),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
