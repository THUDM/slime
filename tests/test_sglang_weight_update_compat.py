"""Legacy bucket updates and explicit SGLang sessions must both remain usable."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

NUM_GPUS = 0


@pytest.fixture(autouse=True)
def requires_weight_update_sessions():
    from sglang.srt.managers import io_struct

    if not hasattr(io_struct, "BeginWeightUpdateReqInput"):
        pytest.skip("These migration checks require SGLang's explicit weight-update sessions")


@pytest.fixture
def manager(monkeypatch):
    from sglang.srt.managers.scheduler_components import weight_updater

    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: None)
    runners = []
    for _ in range(3):
        updater = SimpleNamespace(
            begin_weight_update=Mock(),
            end_weight_update=Mock(),
            update_weights_from_tensor=Mock(return_value=(True, "Success")),
            load_weights_from_distributed=Mock(return_value=(True, "Success")),
            receive_weights_from_distributed=Mock(return_value=[("weight", torch.ones(2))]),
        )
        runners.append(
            SimpleNamespace(weight_updater=updater, post_process_weights=Mock(return_value=(True, "Success")))
        )
    target, *drafts = runners
    worker = SimpleNamespace(
        model_runner=target,
        weight_update_runners=lambda: [("target", target)],
        deserialize_own_rank=lambda tensors: tensors,
    )
    draft_worker = SimpleNamespace(weight_update_runners=lambda: [(f"draft_{i}", r) for i, r in enumerate(drafts)])
    return (
        weight_updater.SchedulerWeightUpdaterManager(
            tp_worker=worker,
            draft_worker=draft_worker,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=Mock(return_value=True),
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(record_weight_version_change=Mock()),
        ),
        runners,
    )


@pytest.mark.parametrize("transport", ["tensor", "distributed"])
@pytest.mark.parametrize("explicit_session", [False, True])
def test_bucket_updates_with_legacy_or_explicit_session(manager, transport, explicit_session):
    from sglang.srt.managers.io_struct import (
        BeginWeightUpdateReqInput,
        EndWeightUpdateReqInput,
        UpdateWeightsFromDistributedReqInput,
        UpdateWeightsFromTensorReqInput,
    )

    updater, runners = manager
    if explicit_session:
        assert updater.begin_weight_update(BeginWeightUpdateReqInput()).success

    for version in ("1", "2"):
        if transport == "tensor":
            request = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[], flush_cache=False, weight_version=version
            )
            response = updater.update_weights_from_tensor(request)
        else:
            request = UpdateWeightsFromDistributedReqInput(
                names=["weight"], dtypes=["float32"], shapes=[[2]], flush_cache=False, weight_version=version
            )
            response = updater.update_weights_from_distributed(request)
        assert response.success, response.message
        updater.scheduler.record_weight_version_change.assert_called_with(new_version=version)

    method = "update_weights_from_tensor" if transport == "tensor" else "load_weights_from_distributed"
    for runner in runners:
        assert getattr(runner.weight_updater, method).call_count == 2
    updater.flush_cache.assert_not_called()
    if explicit_session:
        assert updater.end_weight_update(EndWeightUpdateReqInput()).success
        for runner in runners:
            runner.weight_updater.begin_weight_update.assert_called_once_with()
            runner.weight_updater.end_weight_update.assert_called_once_with(run_post_load=False)
    else:
        for runner in runners:
            runner.weight_updater.begin_weight_update.assert_not_called()
            runner.weight_updater.end_weight_update.assert_not_called()


def test_legacy_post_process_covers_target_and_every_draft_runner(manager):
    from sglang.srt.managers.io_struct import PostProcessWeightsReqInput

    updater, runners = manager
    response = updater.post_process_weights(PostProcessWeightsReqInput(post_process_quantization=True))
    assert response.success, response.message
    for runner in runners:
        runner.post_process_weights.assert_called_once_with(
            restore_weights_before_load=False, post_process_quantization=True
        )


def test_legacy_disable_draft_model_keeps_draft_weights_unchanged(manager):
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

    updater, (target, *drafts) = manager
    response = updater.update_weights_from_tensor(
        UpdateWeightsFromTensorReqInput(serialized_named_tensors=[], disable_draft_model=True, flush_cache=False)
    )
    assert response.success, response.message
    target.weight_updater.update_weights_from_tensor.assert_called_once()
    for draft in drafts:
        draft.weight_updater.update_weights_from_tensor.assert_not_called()


@pytest.mark.parametrize("success", [True, False])
def test_legacy_post_process_accepts_json_body(monkeypatch, success):
    from fastapi.testclient import TestClient
    from sglang.srt.entrypoints import http_server

    process = AsyncMock(return_value=(success, "processed"))
    monkeypatch.setattr(
        http_server, "_global_state", SimpleNamespace(tokenizer_manager=SimpleNamespace(post_process_weights=process))
    )
    response = TestClient(http_server.app).post(
        "/post_process_weights", json={"restore_weights_before_load": False, "post_process_quantization": True}
    )
    assert response.status_code == (200 if success else 400)
    assert response.json() == {"success": success, "message": "processed"}
    process.assert_awaited_once()
    request = process.call_args.args[0]
    assert request.restore_weights_before_load is False
    assert request.post_process_quantization is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
