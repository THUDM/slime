import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime.observability.rollout_data_utils import tensorize_rollout_data_for_training
from slime.utils.routed_experts import (
    RoutedExpertsMicrobatch,
    RoutedExpertsMicrobatchPrefetcher,
    cleanup_routed_experts_rollout,
    spill_routed_experts,
)
from slime.utils.tensor_store import DiskTensorRef
from slime.utils.types import Sample

NUM_GPUS = 0


def _args(tmp_path, *, keep=False):
    return SimpleNamespace(
        num_layers=6,
        moe_router_topk=2,
        num_experts=16,
        moe_layer_freq=[0, 0, 0, 1, 1, 1],
        rollout_routed_experts_store_dir=str(tmp_path),
        keep_rollout_routed_experts_files=keep,
    )


def _routes(rows=4):
    routes = torch.zeros((rows, 6, 2), dtype=torch.uint8)
    routes[:, 3:, 1] = 7
    return routes


@pytest.mark.unit
def test_spill_round_trip_preserves_disk_reference_through_tensorize(tmp_path):
    routes = _routes()
    sample = Sample(
        index=9,
        tokens=[0] * 5,
        response_length=1,
        loss_mask=[1],
        rollout_routed_experts=routes,
        status=Sample.Status.COMPLETED,
    )

    result = asyncio.run(spill_routed_experts(_args(tmp_path), sample, rollout_id=3))

    assert isinstance(result.rollout_routed_experts, DiskTensorRef)
    assert result.rollout_routed_experts.validated
    assert torch.equal(result.rollout_routed_experts.load(), routes)
    rollout_data = {"rollout_routed_experts": [result.rollout_routed_experts]}
    tensorize_rollout_data_for_training(rollout_data)
    assert rollout_data["rollout_routed_experts"][0] is result.rollout_routed_experts


@pytest.mark.unit
def test_aborted_sample_stays_resident_for_retry(tmp_path):
    routes = _routes()
    sample = Sample(
        tokens=[0] * 5,
        response_length=1,
        loss_mask=[1],
        rollout_routed_experts=[routes[:2], routes[2:]],
        status=Sample.Status.ABORTED,
    )

    result = asyncio.run(spill_routed_experts(_args(tmp_path), sample, rollout_id=3))

    assert isinstance(result.rollout_routed_experts, list)
    assert torch.equal(result.materialize_rollout_routed_experts(replace=False), routes)
    assert not list(tmp_path.iterdir())


@pytest.mark.unit
def test_spill_relinks_cross_rollout_reference_and_cleanup_respects_keep(tmp_path):
    args = _args(tmp_path)
    ref = DiskTensorRef.write(_routes(), tmp_path / "rollout_00000001" / "original.safetensors", validated=True)
    sample = Sample(index=1, rollout_routed_experts=ref, status=Sample.Status.COMPLETED)

    asyncio.run(spill_routed_experts(args, sample, rollout_id=2))

    assert isinstance(sample.rollout_routed_experts, DiskTensorRef)
    assert Path(sample.rollout_routed_experts.path).parent.name == "rollout_00000002"
    assert torch.equal(sample.rollout_routed_experts.load(), _routes())
    cleanup_routed_experts_rollout(args, 2)
    assert not (tmp_path / "rollout_00000002").exists()

    keep_args = _args(tmp_path, keep=True)
    asyncio.run(
        spill_routed_experts(
            keep_args, Sample(index=2, rollout_routed_experts=_routes(), status=Sample.Status.COMPLETED), rollout_id=4
        )
    )
    cleanup_routed_experts_rollout(keep_args, 4)
    assert (tmp_path / "rollout_00000004").exists()


@pytest.mark.unit
def test_lazy_prefetch_reloads_between_forward_and_backward(monkeypatch, tmp_path):
    from slime.utils import routed_experts as routed_experts_module

    ref = DiskTensorRef.write(_routes(rows=3), tmp_path / "routes.safetensors", validated=True)
    prefetcher = RoutedExpertsMicrobatchPrefetcher(prefetch_microbatches=1)
    source = RoutedExpertsMicrobatch([ref], [torch.arange(4)], consumer_count=2, prepare_kwargs={})
    prefetcher.add(source)
    loads = []

    def load_and_prepare():
        tensor = ref.load()
        nbytes = tensor.numel() * tensor.element_size()
        source._loaded_nbytes = nbytes
        prefetcher.on_loaded(nbytes)
        loads.append(1)
        return tensor

    monkeypatch.setattr(source, "_load_and_prepare", load_and_prepare)
    monkeypatch.setattr(routed_experts_module.accelerator, "current_device", lambda: torch.device("cpu"))

    prefetcher.begin_pass("forward")
    source.layer_to_cuda(3, "forward")
    source.layer_to_cuda(4, "forward")
    assert prefetcher._resident_bytes == 0

    prefetcher.begin_pass("backward")
    source.layer_to_cuda(3, "backward")
    source.layer_to_cuda(4, "backward")
    assert prefetcher._resident_bytes == 0
    prefetcher.close()
    assert len(loads) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
