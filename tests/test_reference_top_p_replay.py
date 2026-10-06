"""CPU regression for reference KL versus actor top-p replay."""

import ast
import math
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from megatron.core import mpu
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy


NUM_GPUS = 0


def _actor_methods():
    source = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/actor.py"
    actor = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    return {node.name: node for node in actor.body if isinstance(node, ast.FunctionDef)}


@pytest.mark.unit
def test_reference_log_probs_use_full_vocab_while_actor_replays_top_p(monkeypatch):
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)

    def fake_forward_only(
        callback, args, model, data_iterator, _num_microbatches, *, store_prefix, use_rollout_top_p_replay
    ):
        kwargs = {}
        if use_rollout_top_p_replay:
            kwargs = {"top_p_token_ids": [[0, 1]], "top_p_token_offsets": [[0, 2]]}
        _, result = callback(
            model,
            args=args,
            unconcat_tokens=data_iterator,
            total_lengths=[2],
            response_lengths=[1],
            **kwargs,
        )
        return {store_prefix + key: value for key, value in result.items()}

    # Isolate the real actor method: importing the whole actor requires Ray and
    # GPU Megatron modules that are not needed for this CPU probability check.
    method = _actor_methods()["compute_log_prob"]
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {
        "timer": lambda _name: nullcontext(),
        "forward_only": fake_forward_only,
        "get_log_probs_and_entropy": get_log_probs_and_entropy,
    }
    exec(compile(ast.fix_missing_locations(isolated), "actor.py", "exec"), namespace)
    compute_log_prob = namespace["compute_log_prob"]

    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        log_probs_chunk_size=-1,
        entropy_coef=0.0,
        use_rollout_entropy=False,
    )
    tokens = [torch.tensor([0, 0])]

    def log_prob(probs, *, replay, prefix=""):
        logits = torch.tensor([probs, probs], dtype=torch.float32).log().unsqueeze(0)
        actor = SimpleNamespace(args=args, model=logits)
        replay_option = {} if replay is None else {"use_rollout_top_p_replay": replay}
        result = compute_log_prob(actor, tokens, [1], store_prefix=prefix, **replay_option)
        return result[prefix + "log_probs"][0].item()

    actor_log_prob = log_prob([0.2, 0.2, 0.6], replay=None)
    full_ref_a = log_prob([0.1, 0.2, 0.7], replay=False, prefix="ref_")
    full_ref_b = log_prob([0.2, 0.4, 0.4], replay=False, prefix="ref_")
    replay_ref_a = log_prob([0.1, 0.2, 0.7], replay=True, prefix="ref_")
    replay_ref_b = log_prob([0.2, 0.4, 0.4], replay=True, prefix="ref_")

    assert math.exp(actor_log_prob) == pytest.approx(0.5)
    assert math.exp(full_ref_a) == pytest.approx(0.1)
    assert math.exp(full_ref_b) == pytest.approx(0.2)
    assert math.exp(replay_ref_a) == pytest.approx(1 / 3)
    assert replay_ref_b == pytest.approx(replay_ref_a)
    assert actor_log_prob - full_ref_a == pytest.approx(math.log(5))
    assert actor_log_prob - replay_ref_a == pytest.approx(math.log(1.5))

    # The training path must request full normalization for ref_log_probs.
    train_actor = _actor_methods()["train_actor"]
    reference_calls = [
        node
        for node in ast.walk(train_actor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute_log_prob"
        and any(
            keyword.arg == "store_prefix" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "ref_"
            for keyword in node.keywords
        )
    ]
    assert len(reference_calls) == 1
    assert any(
        keyword.arg == "use_rollout_top_p_replay"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in reference_calls[0].keywords
    )
