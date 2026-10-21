"""Lazy R3 reads preserve every CP/TP layout without full-sample materialization."""

import _cp_dist_helpers  # noqa: F401 - install the CPU topology stub first
import pytest
import torch
from straw import SharedFilesystemStore
from straw.tensor import TensorRef, publish_tensors

from slime.backends.megatron_utils import cp_utils
from slime.backends.megatron_utils.cp_utils import prepare_routed_experts_for_routing_replay
from slime.utils.routed_experts import RoutedExpertsMicrobatch, RoutedExpertsMicrobatchPrefetcher

NUM_GPUS = 0


def _reference_pad_routed_experts(experts: torch.Tensor, pad: int, num_experts: int) -> torch.Tensor:
    if pad == 0:
        return experts
    _, num_layers, topk = experts.shape
    values = (torch.arange(pad * num_layers * topk, dtype=torch.int64) % num_experts).to(experts.dtype)
    pad_experts = values.reshape((pad, num_layers, topk))
    return torch.cat([experts, pad_experts], dim=0)


def _reference_slice_routed_experts_with_cp(experts: torch.Tensor, cp_size: int, cp_rank: int, num_experts: int):
    if cp_size == 1:
        return experts

    token_len = len(experts)
    chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    pad = 2 * cp_size * chunk_size - token_len
    experts = _reference_pad_routed_experts(experts, pad, num_experts)

    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([experts[start_1:end_1], experts[start_2:end_2]], dim=0)


def _reference_prepare_routed_experts(
    rollout_routed_experts: list[torch.Tensor],
    tokens: list[torch.Tensor],
    *,
    num_experts: int,
    data_pad_size_multiplier: int,
    sequence_parallel: bool,
    allgather_cp: bool,
    cp_size: int,
    cp_rank: int,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    padded_experts = [_reference_pad_routed_experts(experts, 1, num_experts) for experts in rollout_routed_experts]
    pad_size = tp_size * data_pad_size_multiplier

    if allgather_cp:
        routed_experts = torch.cat(padded_experts, dim=0)
        global_pad_size = cp_size * pad_size
        pad = (global_pad_size - routed_experts.size(0) % global_pad_size) % global_pad_size
        routed_experts = _reference_pad_routed_experts(routed_experts, pad, num_experts)
        routed_experts = routed_experts.chunk(cp_size, dim=0)[cp_rank]
    else:
        routed_experts = [
            _reference_slice_routed_experts_with_cp(experts, cp_size, cp_rank, num_experts)
            for experts in padded_experts
        ]
        routed_experts = torch.cat(routed_experts, dim=0)
        pad = (pad_size - routed_experts.size(0) % pad_size) % pad_size
        routed_experts = _reference_pad_routed_experts(routed_experts, pad, num_experts)

    if sequence_parallel:
        seqlen = routed_experts.size(0)
        assert seqlen % tp_size == 0
        start = seqlen // tp_size * tp_rank
        end = seqlen // tp_size * (tp_rank + 1)
        routed_experts = routed_experts[start:end]

    return routed_experts


@pytest.fixture(autouse=True)
def cpu_allocations(monkeypatch):
    # This CPU-only equivalence test cannot allocate CUDA-pinned host memory.
    empty = torch.empty

    def unpinned(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", unpinned)


@pytest.mark.parametrize("backend", ["straw", "mixed"])
@pytest.mark.parametrize("cp_size", [1, 2, 32])
@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8])
def test_selective_reads_match_complete_reference(
    tmp_path, monkeypatch, backend, cp_size, allgather_cp, sequence_parallel, dtype
):
    num_experts = 288 if dtype == torch.int32 else 251
    tensors = [
        (torch.arange(length * 6 * 2).reshape(length, 6, 2) % num_experts).to(dtype) for length in [0, 1, 7, 127, 256]
    ]
    tokens = [torch.arange(len(x) + 1) for x in tensors]
    with SharedFilesystemStore(tmp_path / "straw", "cp-test", codecs=("tensor.v1",)) as store:
        refs = publish_tensors(store, {str(i): x for i, x in enumerate(tensors)}, submission_id="r3")
    if backend == "mixed":
        refs = [x if i % 2 else tensors[i] for i, x in enumerate(refs)]
    monkeypatch.setattr(TensorRef, "load", lambda *a, **k: pytest.fail("full straw tensor load"))
    mpu = cp_utils.mpu
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)
    for cp_rank in range(cp_size):
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda rank=cp_rank: rank)
        for tp_rank in range(2 if sequence_parallel else 1):
            monkeypatch.setattr(
                mpu,
                "get_tensor_model_parallel_rank",
                lambda rank=tp_rank: rank,
                raising=False,
            )
            kwargs = dict(
                num_experts=num_experts,
                data_pad_size_multiplier=2,
                sequence_parallel=sequence_parallel,
                allgather_cp=allgather_cp,
            )
            actual = prepare_routed_experts_for_routing_replay(refs, tokens, **kwargs)
            expected = _reference_prepare_routed_experts(
                tensors,
                tokens,
                cp_size=cp_size,
                cp_rank=cp_rank,
                tp_size=2,
                tp_rank=tp_rank,
                **kwargs,
            )
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("resident_type", ["tensor", "numpy", "list"])
def test_microbatch_passes_lazy_refs_to_layout_preparation(tmp_path, monkeypatch, resident_type):
    routes = torch.arange(17 * 6 * 2, dtype=torch.int32).reshape(17, 6, 2) % 288
    with SharedFilesystemStore(tmp_path, "microbatch", codecs=("tensor.v1",)) as store:
        (ref,) = publish_tensors(store, {"routes": routes}, submission_id="r3")
    monkeypatch.setattr(TensorRef, "load", lambda *a, **k: pytest.fail("full tensor load"))
    mpu = cp_utils.mpu
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 32)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    kwargs = dict(
        num_experts=288,
        data_pad_size_multiplier=2,
        sequence_parallel=False,
        allgather_cp=True,
    )
    resident = {"tensor": routes, "numpy": routes.numpy(), "list": routes.tolist()}[resident_type]
    for rank in (0, 16, 31):
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda rank=rank: rank)
        source = RoutedExpertsMicrobatch(
            [ref, resident],
            [torch.arange(18), torch.arange(18)],
            consumer_count=6,
            prepare_kwargs=kwargs,
        )
        actual = source._load_and_prepare()
        expected = _reference_prepare_routed_experts(
            [routes, torch.as_tensor(resident)],
            [torch.arange(18), torch.arange(18)],
            cp_size=32,
            cp_rank=rank,
            tp_size=1,
            tp_rank=0,
            **kwargs,
        )
        assert torch.equal(actual, expected)


def test_prefetch_propagates_corrupt_requested_route_bytes(tmp_path, monkeypatch):
    import struct
    from pathlib import Path

    pytest.importorskip("straw")
    from straw import SharedFilesystemStore
    from straw.errors import CorruptData
    from straw.tensor import publish_tensors

    with SharedFilesystemStore(tmp_path, "corrupt-r3", codecs=("tensor.v1",)) as store:
        routes = torch.arange(17 * 6 * 2, dtype=torch.int32).reshape(17, 6, 2) % 288
        (ref,) = publish_tensors(store, {"routes": routes}, submission_id="r3")
        record = store.inspect_segment(ref.record_ref.segment)["records"][ref.record_ref.ordinal]
        start = ref.record_ref.segment.offset + record["offset"]
        with Path(ref.path).open("r+b") as stream:
            stream.seek(start)
            envelope_size, _ = struct.unpack("<IQ", stream.read(12))
            stream.seek(start + 12 + envelope_size)
            original = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([original[0] ^ 1]))
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 32)
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda: 0)
        monkeypatch.setattr(
            cp_utils.mpu,
            "get_tensor_model_parallel_world_size",
            lambda: 1,
            raising=False,
        )
        kwargs = dict(
            num_experts=288,
            data_pad_size_multiplier=2,
            sequence_parallel=False,
            allgather_cp=True,
        )
        source = RoutedExpertsMicrobatch([ref], [torch.arange(18)], consumer_count=6, prepare_kwargs=kwargs)
        prefetcher = RoutedExpertsMicrobatchPrefetcher(1)
        prefetcher.add(source)
        try:
            prefetcher.start()
            with pytest.raises(CorruptData, match="chunk checksum"):
                source._get_cpu_tensor()
        finally:
            prefetcher.close()
        assert source._cpu_tensor is None
        assert source._future is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
