from argparse import Namespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch
from _cp_dist_helpers import cp_chunk_response_tensor, free_port, init_worker_process_group, stub_megatron_in_worker

from slime.backends.megatron_utils.train_dump_utils import (
    _build_dump_payload,
    restore_context_parallel_fields_to_cpu,
    save_debug_train_data,
)

NUM_GPUS = 0


def _patch_single_dp_writer(monkeypatch, mpu, *, cp_size, writer_rank):
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: None, raising=False)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda **_kwargs: True, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(
        mpu,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 0,
        raising=False,
    )
    monkeypatch.setattr(
        mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 1,
        raising=False,
    )
    monkeypatch.setattr(
        mpu,
        "get_data_parallel_src_rank",
        lambda with_context_parallel=False: writer_rank,
        raising=False,
    )


def _restore_context_parallel_worker(rank, world_size, master_port, result_path):
    import torch.distributed as dist
    import torch.distributed.nn  # noqa: F401

    stub_megatron_in_worker(cp_size=world_size, cp_rank=rank)
    cp_group = init_worker_process_group(rank, world_size, master_port)
    try:
        from megatron.core import mpu
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        mpu.get_context_parallel_group = lambda: cp_group

        full_log_probs = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
        local_log_probs = cp_chunk_response_tensor(full_log_probs, total_length=8, response_length=6)
        restored = restore_context_parallel_fields_to_cpu(
            {
                "total_lengths": [8],
                "response_lengths": [6],
                "log_probs": [local_log_probs],
            },
            all_gather_with_cp,
            keep_restored=rank == 0,
        )
        if rank == 0:
            assert restored is not None
            assert restored["log_probs"][0].device.type == "cpu"
            torch.save(restored["log_probs"][0], result_path)
        else:
            assert restored is None
    finally:
        dist.destroy_process_group()


def _single_file_dump_worker(rank, world_size, master_port, result_path):
    import torch.distributed as dist
    import torch.distributed.nn  # noqa: F401

    cp_size = 2
    cp_rank = rank % cp_size
    dp_rank = rank // cp_size
    stub_megatron_in_worker(cp_size=cp_size, cp_rank=cp_rank)
    init_worker_process_group(rank, world_size, master_port)
    cp_groups = [dist.new_group(ranks=[0, 1]), dist.new_group(ranks=[2, 3])]
    dp_group = dist.new_group(ranks=[0, 2])
    try:
        from megatron.core import mpu

        mpu.get_context_parallel_group = lambda: cp_groups[dp_rank]
        mpu.is_pipeline_last_stage = lambda **_kwargs: True
        mpu.get_tensor_model_parallel_rank = lambda: 0
        mpu.get_data_parallel_rank = lambda with_context_parallel=False: dp_rank
        mpu.get_data_parallel_world_size = lambda with_context_parallel=False: 2
        mpu.get_data_parallel_src_rank = lambda with_context_parallel=False: 0
        mpu.get_data_parallel_group_gloo = lambda with_context_parallel=False: dp_group

        full_log_probs = torch.arange(6, dtype=torch.float32) + dp_rank * 10
        local_log_probs = cp_chunk_response_tensor(full_log_probs, total_length=8, response_length=6)
        # Interleave sample_index across DP ranks (dp0 -> 1, dp1 -> 0) so the
        # writer must sort by sample_index to restore global rollout order.
        global_index = {0: 1, 1: 0}[dp_rank]
        save_debug_train_data(
            Namespace(save_debug_train_data=result_path),
            rollout_id=7,
            rollout_data={
                "tokens": [torch.arange(8) + dp_rank * 100],
                "total_lengths": [8],
                "response_lengths": [6],
                "sample_indices": [global_index],
                "log_probs": [local_log_probs],
                "micro_batch_indices": [[[0]]],
            },
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_restore_context_parallel_fields_with_real_collective(tmp_path):
    import torch.multiprocessing as mp

    result_path = str(tmp_path / "restored.pt")
    mp.spawn(
        _restore_context_parallel_worker,
        args=(2, free_port(), result_path),
        nprocs=2,
        join=True,
    )

    torch.testing.assert_close(
        torch.load(result_path, weights_only=True),
        torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]),
    )


def test_save_debug_train_data_writes_one_file_with_all_dp_shards(tmp_path):
    import torch.multiprocessing as mp

    result_path = str(tmp_path / "single.pt")
    mp.spawn(
        _single_file_dump_worker,
        args=(4, free_port(), result_path),
        nprocs=4,
        join=True,
    )

    assert [path.name for path in tmp_path.iterdir()] == ["single.pt"]
    saved = torch.load(result_path, weights_only=True)
    assert saved["format_version"] == 2
    assert saved["rollout_id"] == 7
    assert saved["rank"] == 0
    # samples are sorted by sample_index across DP shards -> global rollout order.
    assert [sample["sample_index"] for sample in saved["samples"]] == [0, 1]
    assert [sample["data_parallel_rank"] for sample in saved["samples"]] == [1, 0]
    # sample_index 0 came from dp_rank 1 (arange + 10), sample_index 1 from dp_rank 0.
    torch.testing.assert_close(saved["samples"][0]["log_probs"], torch.arange(6, dtype=torch.float32) + 10)
    torch.testing.assert_close(saved["samples"][1]["log_probs"], torch.arange(6, dtype=torch.float32))
    # The parallel dp_shards key keeps the DP/mbs layout without duplicating tensors.
    assert [shard["rank"] for shard in saved["dp_shards"]] == [0, 2]
    assert [shard["data_parallel_rank"] for shard in saved["dp_shards"]] == [0, 1]
    assert [shard["sample_indices"] for shard in saved["dp_shards"]] == [[1], [0]]
    assert saved["dp_shards"][0]["micro_batch_indices"] == [[[0]]]
    assert "log_probs" not in saved["dp_shards"][0]


def test_save_debug_train_data_restores_cp_fields_by_default(tmp_path, monkeypatch):
    from megatron.core import mpu
    from slime.backends.megatron_utils import cp_utils

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    _patch_single_dp_writer(monkeypatch, mpu, cp_size=2, writer_rank=3)

    calls = []

    def gather_tensor(value, total_length, response_length):
        calls.append((value.clone(), total_length, response_length))
        return torch.arange(response_length, dtype=torch.float32) + total_length

    monkeypatch.setattr(cp_utils, "all_gather_with_cp", gather_tensor)
    path_template = str(tmp_path / "train_{rollout_id}_{rank}.pt")
    args = Namespace(save_debug_train_data=path_template)
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21, 22, 23, 24])],
        "total_lengths": [4, 5],
        "response_lengths": [2, 3],
        "sample_indices": [0, 1],
        "loss_masks": [torch.tensor([1, 1]), torch.tensor([1, 0, 1])],
        "log_probs": [torch.tensor([-0.1]), torch.tensor([-0.2, -0.3])],
        "advantages": [torch.tensor([0.5]), torch.tensor([0.6, 0.7])],
        "rewards": [1.0, 0.0],
    }

    save_debug_train_data(args, rollout_id=9, rollout_data=rollout_data)

    saved = torch.load(tmp_path / "train_9_3.pt", weights_only=True)
    assert set(saved) == {"format_version", "rollout_id", "rank", "samples", "dp_shards"}
    assert saved["format_version"] == 2
    assert saved["rollout_id"] == 9
    assert saved["rank"] == 3
    assert [sample["sample_index"] for sample in saved["samples"]] == [0, 1]
    assert [sample["data_parallel_rank"] for sample in saved["samples"]] == [0, 0]
    assert [sample["log_probs"].tolist() for sample in saved["samples"]] == [[4.0, 5.0], [5.0, 6.0, 7.0]]
    assert [sample["advantages"].tolist() for sample in saved["samples"]] == [[4.0, 5.0], [5.0, 6.0, 7.0]]
    assert [sample["rewards"] for sample in saved["samples"]] == rollout_data["rewards"]
    assert saved["dp_shards"][0]["sample_indices"] == [0, 1]
    # The source rollout_data tensors are left untouched (writer works on CPU copies).
    torch.testing.assert_close(rollout_data["log_probs"][0], torch.tensor([-0.1]))
    torch.testing.assert_close(rollout_data["log_probs"][1], torch.tensor([-0.2, -0.3]))
    assert [(total, response) for _, total, response in calls] == [(4, 2), (5, 3), (4, 2), (5, 3)]


def test_save_debug_train_data_without_cp_uses_same_normalized_format(tmp_path, monkeypatch):
    from megatron.core import mpu

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    _patch_single_dp_writer(monkeypatch, mpu, cp_size=1, writer_rank=0)
    args = Namespace(
        save_debug_train_data=str(tmp_path / "no_cp_{rollout_id}_{rank}.pt"),
    )
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 12, 13])],
        "total_lengths": [4],
        "response_lengths": [2],
        "sample_indices": [0],
        "log_probs": [torch.tensor([-0.1, -0.2], requires_grad=True)],
        "advantages": [torch.tensor([0.5, 0.6])],
    }

    save_debug_train_data(args, rollout_id=2, rollout_data=rollout_data)

    saved = torch.load(tmp_path / "no_cp_2_0.pt", weights_only=True)
    assert set(saved) == {"format_version", "rollout_id", "rank", "samples", "dp_shards"}
    assert len(saved["samples"]) == 1
    sample = saved["samples"][0]
    assert sample["sample_index"] == 0
    for key in ("log_probs", "advantages"):
        assert len(sample[key]) == sample["response_lengths"]
        assert sample[key].device.type == "cpu"
        assert not sample[key].requires_grad
        torch.testing.assert_close(sample[key], rollout_data[key][0])


def _layout_shard(rank, dp_rank, sample_indices, log_probs, **extra):
    return {
        "rank": rank,
        "data_parallel_rank": dp_rank,
        "rollout_data": {
            "response_lengths": [len(lp) for lp in log_probs],
            "sample_indices": sample_indices,
            "log_probs": log_probs,
            **extra,
        },
    }


def test_build_dump_payload_sorts_samples_and_keeps_layout():
    dp_shards = [
        _layout_shard(
            0,
            0,
            [2, 0],
            [torch.tensor([2.0]), torch.tensor([0.0])],
            micro_batch_indices=[[0, 1]],
            num_microbatches=[1],
            global_batch_sizes=[1],
            raw_reward=[9.0],
        ),
        _layout_shard(
            2,
            1,
            [3, 1],
            [torch.tensor([3.0]), torch.tensor([1.0])],
            micro_batch_indices=[[0, 1]],
            num_microbatches=[1],
            global_batch_sizes=[1],
            raw_reward=[9.0],
        ),
    ]

    payload = _build_dump_payload(dp_shards, rollout_id=5, writer_rank=0)

    assert payload["format_version"] == 2
    # samples flattened across shards and sorted by global sample_index.
    assert [sample["sample_index"] for sample in payload["samples"]] == [0, 1, 2, 3]
    assert [sample["log_probs"].item() for sample in payload["samples"]] == [0.0, 1.0, 2.0, 3.0]
    assert [sample["data_parallel_rank"] for sample in payload["samples"]] == [0, 1, 0, 1]
    # per-rank scheduling stays in the parallel dp_shards key, not in samples.
    assert "micro_batch_indices" not in payload["samples"][0]
    assert [shard["sample_indices"] for shard in payload["dp_shards"]] == [[2, 0], [3, 1]]
    assert payload["dp_shards"][0]["micro_batch_indices"] == [[0, 1]]
    assert payload["dp_shards"][0]["num_microbatches"] == [1]
    # whole-batch fields are stored once at the top level.
    assert payload["raw_reward"] == [9.0]


def test_build_dump_payload_keeps_gather_order_when_index_missing():
    dp_shards = [
        _layout_shard(0, 0, None, [torch.tensor([2.0]), torch.tensor([0.0])]),
    ]

    payload = _build_dump_payload(dp_shards, rollout_id=1, writer_rank=0)

    # No sample_index available -> keep DP-gather order, no sample_index key.
    assert [sample["log_probs"].item() for sample in payload["samples"]] == [2.0, 0.0]
    assert "sample_index" not in payload["samples"][0]
    assert payload["dp_shards"][0]["sample_indices"] is None


def test_build_dump_payload_uses_rollout_positions_when_sample_index_missing():
    # sample_indices all None, but rollout_positions (DP partition) restores the
    # exact rollout-dump order regardless of sample.index.
    dp_shards = [
        {
            "rank": 0,
            "data_parallel_rank": 0,
            "rollout_data": {
                "response_lengths": [1, 1],
                "sample_indices": [None, None],
                "rollout_positions": [2, 0],
                "log_probs": [torch.tensor([2.0]), torch.tensor([0.0])],
            },
        },
        {
            "rank": 2,
            "data_parallel_rank": 1,
            "rollout_data": {
                "response_lengths": [1, 1],
                "sample_indices": [None, None],
                "rollout_positions": [3, 1],
                "log_probs": [torch.tensor([3.0]), torch.tensor([1.0])],
            },
        },
    ]

    payload = _build_dump_payload(dp_shards, rollout_id=8, writer_rank=0)

    assert [sample["rollout_position"] for sample in payload["samples"]] == [0, 1, 2, 3]
    assert [sample["log_probs"].item() for sample in payload["samples"]] == [0.0, 1.0, 2.0, 3.0]
    assert [shard["rollout_positions"] for shard in payload["dp_shards"]] == [[2, 0], [3, 1]]


def test_restore_context_parallel_fields_rejects_misaligned_samples():
    rollout_data = {
        "total_lengths": [4, 5],
        "response_lengths": [2, 3],
        "log_probs": [torch.tensor([-0.1])],
    }

    with pytest.raises(ValueError, match="one tensor per sample"):
        restore_context_parallel_fields_to_cpu(
            rollout_data,
            lambda value, _total, _response: value,
            keep_restored=True,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
