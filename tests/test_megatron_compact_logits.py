import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0


def has_config_logger_enabled(config) -> bool:
    return bool(config.config_logger_enabled)


def _run_output_layer(self, hidden_states, runtime_gather_output):
    logits, _ = self.output_layer(
        hidden_states,
        weight=None,
        runtime_gather_output=runtime_gather_output,
    )
    return logits.transpose(0, 1).contiguous()


def _postprocess_with_mtp_kwargs(
    self,
    hidden_states,
    input_ids=None,
    position_ids=None,
    labels=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    mtp_in_postprocess=None,
    loss_mask=None,
    decoder_input=None,
    attention_mask=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
    mtp_kwargs=None,
):
    return _run_output_layer(self, hidden_states, runtime_gather_output)


def _postprocess_without_mtp_kwargs(
    self,
    hidden_states,
    input_ids=None,
    position_ids=None,
    labels=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    mtp_in_postprocess=None,
    loss_mask=None,
    decoder_input=None,
    attention_mask=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
):
    return _run_output_layer(self, hidden_states, runtime_gather_output)


class _PostProcessNode:
    def __init__(self, model, hidden_states, loss_mask):
        self.model = model
        self.hidden_states = hidden_states
        self.loss_mask = loss_mask

    def forward_impl(self):
        return self.model._postprocess(hidden_states=self.hidden_states, loss_mask=self.loss_mask)


def _build_schedule_plan(self, hidden_states, loss_mask):
    return _PostProcessNode(self, hidden_states, loss_mask)


class _Group:
    def __init__(self, size=1):
        self._size = size

    def size(self):
        return self._size


class _ColumnParallelLinear(torch.nn.Module):
    def __init__(self, hidden_size=3, vocab_size=5, *, sequence_parallel=False, tp_size=1):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(vocab_size, hidden_size) / 10
        )
        self.bias = None
        self.output_size_per_partition = vocab_size
        self.sequence_parallel = sequence_parallel
        self.gather_output = False
        self.allreduce_dgrad = False
        self.disable_grad_reduce = False
        self.explicit_expert_comm = False
        self.tp_group = _Group(tp_size)
        self.calls = []
        self.raise_on_forward = False

    def forward(self, input_, weight=None, runtime_gather_output=None):
        self.calls.append(input_.size(0))
        if self.raise_on_forward:
            raise RuntimeError("projection failed")
        if self.sequence_parallel:
            input_ = sys.modules["megatron.core.tensor_parallel"].gather_from_sequence_parallel_region(
                input_, tensor_parallel_output_grad=True, group=self.tp_group
            )
        weight = self.weight if weight is None else weight
        return torch.nn.functional.linear(input_, weight), None


def _load_compact_logits(monkeypatch, *, with_mtp_kwargs=True):
    gather_calls = []

    def gather(input_, tensor_parallel_output_grad=True, group=None, **_kwargs):
        gather_calls.append((tensor_parallel_output_grad, group))
        if group.size() == 1:
            return input_
        return torch.cat((input_, input_ + 10), dim=0)

    original = _postprocess_with_mtp_kwargs if with_mtp_kwargs else _postprocess_without_mtp_kwargs
    gpt_model = type(
        "GPTModel",
        (torch.nn.Module,),
        {"_postprocess": original, "build_schedule_plan": _build_schedule_plan},
    )

    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.models": types.ModuleType("megatron.core.models"),
        "megatron.core.models.gpt": types.ModuleType("megatron.core.models.gpt"),
        "megatron.core.tensor_parallel": types.ModuleType("megatron.core.tensor_parallel"),
        "megatron.core.tensor_parallel.layers": types.ModuleType("megatron.core.tensor_parallel.layers"),
    }
    modules["megatron.core.models.gpt"].GPTModel = gpt_model
    modules["megatron.core.tensor_parallel"].gather_from_sequence_parallel_region = gather
    modules["megatron.core.tensor_parallel.layers"].ColumnParallelLinear = _ColumnParallelLinear
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "compact_logits.py"
    module_name = f"test_compact_logits_{'mtp' if with_mtp_kwargs else 'legacy'}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, gpt_model, gather_calls, original


def _make_model(compact_logits, *, sequence_parallel=False, tp_size=1):
    model = compact_logits.CompactLogitsGPTModel()
    model.post_process = True
    model.parallel_output = True
    model.share_embeddings_and_output_weights = False
    model.mtp_process = False
    model.config = types.SimpleNamespace(
        sequence_parallel=sequence_parallel,
        mtp_num_layers=None,
        config_logger_enabled=False,
        defer_embedding_wgrad_compute=False,
        cuda_graph_impl="none",
    )
    model.output_layer = _ColumnParallelLinear(sequence_parallel=sequence_parallel, tp_size=tp_size)
    return model


def _call(model, hidden_states, loss_mask, **kwargs):
    return model._postprocess(hidden_states=hidden_states, loss_mask=loss_mask, **kwargs)


@pytest.mark.unit
def test_eligibility_is_opt_in_and_limited_to_builtin_masked_paths(monkeypatch):
    compact_logits, _, _, _ = _load_compact_logits(monkeypatch)
    args = types.SimpleNamespace(compact_actor_logits=True, loss_type="policy_loss")

    assert compact_logits.can_compact_actor_logits(args)
    args.loss_type = "sft_loss"
    assert compact_logits.can_compact_actor_logits(args)
    args.loss_type = "custom_loss"
    assert not compact_logits.can_compact_actor_logits(args)
    args.loss_type = "policy_loss"
    args.compact_actor_logits = False
    assert not compact_logits.can_compact_actor_logits(args)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("enable_mtp_training", True),
        ("custom_model_provider_path", "custom.provider"),
        ("custom_megatron_init_path", "custom.init"),
        ("custom_megatron_before_log_prob_hook_path", "custom.logprob_hook"),
        ("custom_megatron_before_train_step_hook_path", "custom.train_hook"),
        ("custom_advantage_function_path", "custom.advantage"),
        ("rollout_data_postprocess_path", "custom.postprocess"),
        ("custom_pg_loss_reducer_function_path", "custom.reducer"),
        ("use_tis", True),
        ("get_mismatch_metrics", True),
        ("use_rollout_entropy", True),
        ("save_debug_train_data", "debug.pt"),
        ("use_rollout_logprobs", True),
    ],
)
def test_eligibility_falls_back_when_off_mask_values_are_observable(monkeypatch, name, value):
    compact_logits, _, _, _ = _load_compact_logits(monkeypatch)
    args = types.SimpleNamespace(compact_actor_logits=True, loss_type="policy_loss")
    setattr(args, name, value)

    assert not compact_logits.can_compact_actor_logits(args)


@pytest.mark.unit
def test_eligibility_keeps_ppo_reward_kl_full(monkeypatch):
    compact_logits, _, _, _ = _load_compact_logits(monkeypatch)
    args = types.SimpleNamespace(
        compact_actor_logits=True,
        loss_type="policy_loss",
        advantage_estimator="ppo",
        kl_coef=0.1,
    )

    assert not compact_logits.can_compact_actor_logits(args)
    args.kl_coef = 0
    assert compact_logits.can_compact_actor_logits(args)


@pytest.mark.unit
@pytest.mark.parametrize("with_mtp_kwargs", [False, True])
def test_actor_subclass_compacts_without_mutating_base_model(monkeypatch, with_mtp_kwargs):
    compact_logits, gpt_model, _, original = _load_compact_logits(monkeypatch, with_mtp_kwargs=with_mtp_kwargs)
    model = _make_model(compact_logits)
    base_model = gpt_model()
    base_model.post_process = True
    base_model.output_layer = _ColumnParallelLinear()
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)
    mask = torch.tensor([[1, 0, 1, 0]])

    assert gpt_model._postprocess is original
    assert issubclass(compact_logits.CompactLogitsGPTModel, gpt_model)
    with compact_logits.compact_actor_logits(True):
        base_output = _call(base_model, hidden, mask)
        compact_output = _call(model, hidden, mask)

    assert base_output.shape == (1, 4, 5)
    assert compact_output.shape == (1, 2, 5)
    torch.testing.assert_close(compact_output, base_output[:, [0, 2]])


@pytest.mark.unit
def test_context_is_scoped_and_nested(monkeypatch):
    compact_logits, _, _, _ = _load_compact_logits(monkeypatch)
    model = _make_model(compact_logits)
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)
    mask = torch.tensor([[1, 0, 1, 0]])

    full = _call(model, hidden, mask)
    with compact_logits.compact_actor_logits(True):
        compact = _call(model, hidden, mask)
        with compact_logits.compact_actor_logits(False):
            nested_full = _call(model, hidden, mask)
        compact_again = _call(model, hidden, mask)
    restored_full = _call(model, hidden, mask)

    assert full.shape == nested_full.shape == restored_full.shape == (1, 4, 5)
    assert compact.shape == compact_again.shape == (1, 2, 5)
    torch.testing.assert_close(compact, full[:, [0, 2]])


@pytest.mark.unit
def test_combined_schedule_node_dispatches_to_actor_subclass(monkeypatch):
    compact_logits, gpt_model, _, _ = _load_compact_logits(monkeypatch)
    model = _make_model(compact_logits)
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)
    mask = torch.tensor([[1, 0, 1, 0]])
    plan = model.build_schedule_plan(hidden, mask)

    assert compact_logits.CompactLogitsGPTModel.build_schedule_plan is gpt_model.build_schedule_plan
    with compact_logits.compact_actor_logits(True):
        output = plan.forward_impl()

    assert output.shape == (1, 2, 5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate,call_kwargs",
    [
        (lambda model: setattr(model, "post_process", False), {}),
        (lambda model: setattr(model, "parallel_output", False), {}),
        (lambda model: setattr(model.config, "mtp_num_layers", 1), {}),
        (lambda model: setattr(model.config, "config_logger_enabled", True), {}),
        (lambda model: setattr(model.config, "defer_embedding_wgrad_compute", True), {}),
        (lambda model: setattr(model.config, "cuda_graph_impl", "local"), {}),
        (lambda _model: None, {"labels": torch.ones(1, 4, dtype=torch.long)}),
        (lambda _model: None, {"inference_context": object()}),
        (lambda _model: None, {"runtime_gather_output": True}),
    ],
)
def test_unsupported_paths_fall_back_before_projection(monkeypatch, mutate, call_kwargs):
    compact_logits, _, _, _ = _load_compact_logits(monkeypatch)
    model = _make_model(compact_logits)
    mutate(model)
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)

    with compact_logits.compact_actor_logits(True):
        output = _call(model, hidden, torch.tensor([[1, 0, 1, 0]]), **call_kwargs)

    assert output.shape == (1, 4, 5)
    assert model.output_layer.calls == [4]


@pytest.mark.unit
def test_sequence_parallel_gathers_once_without_second_gradient_reduce(monkeypatch):
    compact_logits, _, gather_calls, _ = _load_compact_logits(monkeypatch)
    model = _make_model(compact_logits, sequence_parallel=True, tp_size=2)
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
    mask = torch.tensor([[1, 0, 0, 1]])

    with compact_logits.compact_actor_logits(True):
        output = _call(model, hidden, mask)

    expected_hidden = torch.cat((hidden, hidden + 10), dim=0)[[0, 3]]
    expected = torch.nn.functional.linear(expected_hidden, model.output_layer.weight).transpose(0, 1)
    torch.testing.assert_close(output, expected)
    assert gather_calls == [(False, model.output_layer.tp_group)]
    assert model.output_layer.sequence_parallel is True


@pytest.mark.unit
def test_sequence_parallel_flag_is_restored_after_projection_error(monkeypatch):
    compact_logits, _, _, _ = _load_compact_logits(monkeypatch)
    model = _make_model(compact_logits, sequence_parallel=True, tp_size=2)
    model.output_layer.raise_on_forward = True

    with pytest.raises(RuntimeError, match="projection failed"):
        with compact_logits.compact_actor_logits(True):
            _call(model, torch.ones(2, 1, 3), torch.tensor([[1, 0, 0, 1]]))

    assert model.output_layer.sequence_parallel is True


@pytest.mark.unit
def test_empty_mask_skips_projection_and_keeps_zero_gradient_edges(monkeypatch):
    compact_logits, _, _, _ = _load_compact_logits(monkeypatch)
    model = _make_model(compact_logits)
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3).requires_grad_()

    with compact_logits.compact_actor_logits(True):
        output = _call(model, hidden, torch.zeros(1, 4))

    assert output.shape == (1, 0, 5)
    assert model.output_layer.calls == []
    output.sum().backward()
    assert hidden.grad is not None
    assert model.output_layer.weight.grad is not None
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.count_nonzero(model.output_layer.weight.grad) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
