import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from straw import SharedFilesystemStore
from straw.tensor import publish_tensors

from slime.rollout.sample_hooks import apply_rollout_sample_hooks
from slime.utils.routed_experts import RoutedExpertsMicrobatch, RoutedExpertsMicrobatchPrefetcher
from slime.utils.tensor_store import TensorRef
from slime.utils.types import Sample

NUM_GPUS = 0


def _args(tmp_path):
    return SimpleNamespace(
        rollout_data_transport="object-store",
        num_layers=6,
        moe_router_topk=2,
        num_experts=16,
        moe_layer_freq=[0, 0, 0, 1, 1, 1],
        rollout_data_dir=str(tmp_path),
    )


def _routes(rows=4):
    routes = torch.zeros((rows, 6, 2), dtype=torch.uint8)
    routes[:, 3:, 1] = 7
    return routes


@pytest.mark.parametrize("r3,sc", [(True, False), (False, True), (True, True)])
def test_straw_publishes_replay_tensors_with_the_group_without_hook_or_spill_directory(tmp_path, r3, sc):
    from straw.tensor import TensorRef

    args = _args(tmp_path)
    args.rollout_data_transport = "straw"
    args.rollout_data_dir = str(tmp_path)
    args.use_rollout_routing_replay = r3
    args.use_score_centering = sc
    args.score_centering_top_k = 2
    args.rollout_sample_hook_path = None
    samples = [
        Sample(
            index=i,
            tokens=[0] * 5,
            response_length=1,
            loss_mask=[1],
            rollout_routed_experts=_routes() if r3 else None,
            rollout_topk_token_ids=torch.tensor([[1, 2]], dtype=torch.int32) if sc else None,
            rollout_topk_log_probs=torch.tensor([[0.7, 0.3]]).log() if sc else None,
            status=Sample.Status.COMPLETED,
        )
        for i in range(32)
    ]
    nested = [samples[:16], samples[16:]]
    result = asyncio.run(apply_rollout_sample_hooks(args, nested, rollout_id=3))
    assert result == nested
    assert not list(tmp_path.iterdir())
    from slime.utils.rollout_transport import pack_rollout_payload

    durable = pack_rollout_payload(result, args, 3)
    samples = [sample for group in durable.load() for sample in group]
    for sample in samples:
        if r3:
            assert isinstance(sample.rollout_routed_experts, TensorRef)
            torch.testing.assert_close(sample.rollout_routed_experts.load(), _routes())
        if sc:
            assert isinstance(sample.rollout_topk_token_ids, TensorRef)
            assert isinstance(sample.rollout_topk_log_probs, TensorRef)
            assert sample.rollout_topk_token_ids.load().tolist() == [[1, 2]]
            torch.testing.assert_close(sample.rollout_topk_log_probs.load().exp(), torch.tensor([[0.7, 0.3]]))
    files = {path: path.stat().st_size for path in tmp_path.rglob("*.pack")}
    assert len(files) == 1
    assert not list(tmp_path.rglob("*.safetensors"))
    assert not list(tmp_path.rglob("*.bin"))
    second = pack_rollout_payload([samples[:16], samples[16:]], args, 4)
    restored = [sample for group in second.load() for sample in group]
    for before, after in zip(samples, restored, strict=True):
        for key in ("rollout_routed_experts", "rollout_topk_token_ids", "rollout_topk_log_probs"):
            if getattr(before, key) is not None:
                assert getattr(before, key).record_ref == getattr(after, key).record_ref
    assert len(list(tmp_path.rglob("*.pack"))) == len(files)


@pytest.mark.parametrize(
    "transport,evaluation,status",
    [
        ("object-store", False, Sample.Status.COMPLETED),
        ("straw", True, Sample.Status.COMPLETED),
        ("straw", False, Sample.Status.ABORTED),
    ],
)
def test_hooks_leave_ray_evaluation_and_aborted_samples_resident(tmp_path, transport, evaluation, status):
    args = _args(tmp_path)
    args.rollout_data_transport = transport
    args.rollout_data_dir = str(tmp_path)
    args.use_rollout_routing_replay = True
    sample = Sample(tokens=[0] * 5, rollout_routed_experts=_routes(), status=status)
    routes = sample.rollout_routed_experts
    assert asyncio.run(apply_rollout_sample_hooks(args, sample, evaluation=evaluation)) is sample
    assert sample.rollout_routed_experts is routes
    assert not list(tmp_path.iterdir())


def test_straw_adopts_r3_from_another_pool(tmp_path):
    args = _args(tmp_path)
    args.rollout_data_transport = "straw"
    args.rollout_data_dir = str(tmp_path / "queue")
    args.use_rollout_routing_replay = True
    with SharedFilesystemStore(tmp_path / "source", "source", codecs=("tensor.v1",)) as store:
        (ref,) = publish_tensors(store, {"rollout_routed_experts": _routes()}, submission_id="r3")
    path = Path(ref.path)
    sample = Sample(
        index=1,
        tokens=[0] * 5,
        rollout_routed_experts=ref,
        status=Sample.Status.COMPLETED,
    )
    from slime.utils.rollout_transport import pack_rollout_payload

    sample = pack_rollout_payload(sample, args, 0).load()
    path.unlink()
    torch.testing.assert_close(sample.rollout_routed_experts.load(), _routes())


def test_cancelled_straw_publication_drains_before_releasing_capacity(tmp_path, monkeypatch):
    from slime.utils import rollout_transport

    args = _args(tmp_path)
    args.rollout_data_transport = "straw"
    args.rollout_data_dir = str(tmp_path)
    args.rollout_io_concurrency = 1
    args.use_rollout_routing_replay = True
    samples = [
        Sample(index=i, tokens=[0] * 5, rollout_routed_experts=_routes(), status=Sample.Status.COMPLETED)
        for i in range(2)
    ]
    entered, release = threading.Event(), threading.Event()
    calls = []
    write = rollout_transport.pack_rollout_payload
    published = []

    def delayed_write(*values):
        calls.append(values[0].index)
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release the accepted write")
        result = write(*values)
        published.append(result)
        return result

    monkeypatch.setattr(rollout_transport, "pack_rollout_payload", delayed_write)

    async def run():
        first = asyncio.create_task(rollout_transport.publish_rollout_async(samples[0], args, 0))
        tasks = [first]
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            first.cancel()
            second = asyncio.create_task(rollout_transport.publish_rollout_async(samples[1], args, 0))
            tasks.append(second)
            await asyncio.sleep(0.05)
            first.cancel()  # A second shutdown signal must not release the writer early.
            await asyncio.sleep(0.05)
            assert not first.done()
            assert calls == [0]
        finally:
            release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert results[1] is published[1]
        assert calls == [0, 1]

    asyncio.run(run())
    for result in published:
        sample = result.load()
        assert isinstance(sample.rollout_routed_experts, TensorRef)
        torch.testing.assert_close(sample.rollout_routed_experts.load(), _routes())


@pytest.mark.unit
def test_lazy_prefetch_reloads_between_forward_and_backward(monkeypatch, tmp_path):
    import weakref

    from slime.utils import routed_experts as routed_experts_module

    with SharedFilesystemStore(tmp_path, "prefetch", codecs=("tensor.v1",)) as store:
        (ref,) = publish_tensors(store, {"rollout_routed_experts": _routes(rows=3)}, submission_id="r3")
    prefetcher = RoutedExpertsMicrobatchPrefetcher(prefetch_microbatches=1)
    source = RoutedExpertsMicrobatch([ref], [torch.arange(4)], consumer_count=2, prepare_kwargs={})
    prefetcher.add(source)
    loads = []

    def load_and_prepare():
        tensor = ref.load()
        loads.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(source, "_load_and_prepare", load_and_prepare)
    monkeypatch.setattr(routed_experts_module.accelerator, "current_device", lambda: torch.device("cpu"))

    prefetcher.begin_pass("forward")
    source.layer_to_cuda(3, "forward")
    source.layer_to_cuda(4, "forward")
    assert loads[-1]() is None

    prefetcher.begin_pass("backward")
    source.layer_to_cuda(3, "backward")
    source.layer_to_cuda(4, "backward")
    assert loads[-1]() is None
    prefetcher.close()
    assert len(loads) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
