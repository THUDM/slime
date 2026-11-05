import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch

import _cp_dist_helpers  # noqa: F401


NUM_GPUS = 0


@pytest.mark.unit
def test_triton_labels_and_unpack_align_cp1_response_rows(monkeypatch):
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import build_triton_log_prob_labels, get_log_probs_and_entropy

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    args = Namespace(log_probs_backend="triton", allgather_cp=False)
    tokens = torch.tensor([10, 11, 12, 13, 14, 15, 16])
    batch = {
        "tokens": tokens,
        "unconcat_tokens": [tokens[:4], tokens[4:]],
        "total_lengths": [4, 3],
        "response_lengths": [2, 1],
    }

    assert build_triton_log_prob_labels(args, batch).tolist() == [11, 12, 13, 0, 15, 16, 0]

    fused = torch.tensor(
        [[[0.0, 100.0], [1.0, 101.0], [2.0, 102.0], [3.0, 103.0], [4.0, 104.0], [5.0, 105.0], [6.0, 106.0]]]
    )
    empty, res = get_log_probs_and_entropy(
        fused,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True,
    )

    assert empty.numel() == 0
    assert [x.tolist() for x in res["log_probs"]] == [[1.0, 2.0], [5.0]]
    assert [x.tolist() for x in res["entropy"]] == [[101.0, 102.0], [105.0]]


@pytest.mark.unit
@pytest.mark.parametrize(
    "cp_rank,expected_labels,expected_log_probs",
    [
        (0, [11, 12, 17, 0, 0, 0, 0, 0], [[0.0, 1.0, 2.0], []]),
        (1, [13, 14, 15, 16, 23, 24, 0, 0], [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0]]),
    ],
)
def test_triton_labels_and_unpack_align_cp2_zigzag_response_rows(monkeypatch, cp_rank, expected_labels, expected_log_probs):
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import build_triton_log_prob_labels, get_log_probs_and_entropy

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: cp_rank)
    args = Namespace(log_probs_backend="triton", allgather_cp=False)
    samples = [torch.arange(10, 18), torch.arange(20, 25)]
    batch = {
        "tokens": torch.zeros(8, dtype=torch.long),
        "unconcat_tokens": samples,
        "total_lengths": [8, 5],
        "response_lengths": [7, 2],
    }

    assert build_triton_log_prob_labels(args, batch).tolist() == expected_labels

    fused = torch.stack((torch.arange(8.0), torch.arange(100.0, 108.0)), dim=-1).unsqueeze(0)
    empty, res = get_log_probs_and_entropy(
        fused,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True,
    )

    assert empty.numel() == 0
    assert [x.tolist() for x in res["log_probs"]] == expected_log_probs
    assert [x.tolist() for x in res["entropy"]] == [[x + 100 for x in sample] for sample in expected_log_probs]


@pytest.mark.unit
def test_triton_cp2_empty_rank_keeps_zero_backward(monkeypatch):
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import build_triton_log_prob_labels, get_log_probs_and_entropy

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 1)
    args = Namespace(log_probs_backend="triton", allgather_cp=False)
    samples = [torch.arange(10, 18), torch.arange(20, 28)]
    batch = {
        "tokens": torch.zeros(8, dtype=torch.long),
        "unconcat_tokens": samples,
        "total_lengths": [8, 8],
        "response_lengths": [1, 1],
    }

    assert build_triton_log_prob_labels(args, batch).tolist() == [0] * 8

    fused = torch.randn(1, 8, 2, requires_grad=True)
    empty, res = get_log_probs_and_entropy(
        fused,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True,
    )
    loss = sum(x.sum() for values in res.values() for x in values)
    loss.backward()

    assert empty.numel() == 0
    assert [x.numel() for values in res.values() for x in values] == [0, 0, 0, 0]
    assert fused.grad is not None
    assert torch.isfinite(fused.grad).all()
    assert fused.grad.eq(0).all()


@pytest.mark.unit
def test_install_triton_log_probs_patches_only_postprocess_stage(monkeypatch):
    fake_tensor_parallel = types.SimpleNamespace(
        copy_to_tensor_model_parallel_region=lambda x: x,
        gather_from_sequence_parallel_region=lambda x: x,
    )
    monkeypatch.setattr(sys.modules["megatron.core"], "tensor_parallel", fake_tensor_parallel, raising=False)
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", fake_tensor_parallel)
    module = importlib.reload(importlib.import_module("slime.backends.megatron_utils.triton_log_probs.megatron"))

    def fake_linear_cross_entropy(hidden, weight, labels, temperature, reduction, group):
        assert reduction == "none"
        assert temperature == 0.7
        return labels.float(), labels.float() + 10

    monkeypatch.setattr(module, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(module.mpu, "get_tensor_model_parallel_group", lambda: object(), raising=False)

    class FakeOutputLayer:
        bias = None

        def __init__(self):
            self.weight = torch.ones(5, 3)

    class FakeModel:
        def __init__(self, post_process):
            self.post_process = post_process
            self.config = Namespace(sequence_parallel=False)
            self.share_embeddings_and_output_weights = False
            self.output_layer = FakeOutputLayer()
            self.original_calls = 0

        def _postprocess(self, *args, **kwargs):
            self.original_calls += 1
            return "original"

    first_stage = FakeModel(post_process=False)
    final_stage = FakeModel(post_process=True)
    module.install_triton_log_probs([first_stage, final_stage], Namespace(log_probs_backend="triton", rollout_temperature=0.7))

    assert not hasattr(first_stage, "_slime_triton_log_probs_args")
    assert final_stage._postprocess(labels=None) == "original"
    assert final_stage.original_calls == 1

    out = final_stage._postprocess(hidden_states=torch.ones(2, 3), labels=torch.tensor([3, 4]))
    assert out.shape == (1, 2, 2)
    assert out.tolist() == [[[3.0, 13.0], [4.0, 14.0]]]

    torch_backend = FakeModel(post_process=True)
    module.install_triton_log_probs(
        [torch_backend], Namespace(log_probs_backend="torch", rollout_temperature=0.7)
    )
    assert not hasattr(torch_backend, "_slime_triton_log_probs_args")
    assert torch_backend._postprocess(labels=torch.tensor([1])) == "original"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
