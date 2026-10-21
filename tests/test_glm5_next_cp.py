"""Distributed GLM5-next allgather CP parity, including packed sequence boundaries."""

import copy
import tempfile
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime_plugins.models.glm5_next.attention import KimiDeltaAttention, NopeDSAttention
from slime_plugins.models.glm5_next.model import Glm5NextModel

NUM_GPUS = 4


def test_cp_requires_contiguous_allgather_layout(monkeypatch):
    from slime_plugins.models.glm5_next import glm5_next

    monkeypatch.setattr(glm5_next, "load_hf_config", lambda *args, **kwargs: SimpleNamespace(text_config=None))
    args = SimpleNamespace(hf_checkpoint="unused", allgather_cp=False)
    config = SimpleNamespace(tensor_model_parallel_size=1, context_parallel_size=2)
    with pytest.raises(ValueError, match="requires --allgather-cp"):
        glm5_next.get_glm5_next_spec(args, config, None)


def _assert_relative(actual, expected, name, tolerance=0.02):
    assert actual.isfinite().all(), name
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
    assert error < tolerance, (name, error.item())


def _gate_grad_reference(model, hidden_states, cu, output_grad):
    """Independent FP64 recurrence for the two FP32 KDA gate parameters."""
    with torch.no_grad():
        x = hidden_states.transpose(0, 1)
        qkv = [
            getattr(model, f"{name}_conv1d")(getattr(model, f"{name}_proj")(x), cu_seqlens=cu)[0].unflatten(
                -1, (model.heads, model.head_dim)
            )[0]
            for name in ("q", "k", "v")
        ]
        # Match the kernel's BF16 normalized Q/K inputs before the recurrence.
        q, k = [
            (value.float() * torch.rsqrt(value.float().square().sum(-1, keepdim=True) + 1e-6)).to(value.dtype).double()
            for value in qkv[:2]
        ]
        v = qkv[2].double()
        gate = model.f_b_proj(model.f_a_proj(x)).unflatten(-1, (model.heads, model.head_dim))[0].double()
        beta = model.b_proj(x).sigmoid()[0].double()
    a_log = model.A_log.detach().double().requires_grad_()
    bias = model.dt_bias.detach().double().requires_grad_()
    decay = (model.lower_bound * torch.sigmoid(a_log.exp()[:, None] * (gate + bias.view_as(gate[0])))).exp()
    outputs = []
    boundaries = cu.tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        state = v.new_zeros(model.heads, model.head_dim, model.head_dim)
        for i in range(start, end):
            state = state * decay[i, :, :, None]
            residual = beta[i, :, None] * (v[i] - (k[i, :, :, None] * state).sum(-2))
            state = state + k[i, :, :, None] * residual[:, None, :]
            outputs.append((q[i, :, :, None] * state).sum(-2) * model.head_dim**-0.5)
    grads = torch.autograd.grad(torch.stack(outputs).reshape_as(output_grad), (a_log, bias), output_grad.double())
    return dict(zip(("A_log", "dt_bias"), grads, strict=True))


def _check_attention(rank, world_size, kind, lengths):
    torch.manual_seed(147)
    config = SimpleNamespace(hidden_size=256)
    hf_config = SimpleNamespace(
        linear_attn_config={"num_heads": 4, "head_dim": 128, "safe_gate": True, "short_conv_kernel_size": 4},
        hidden_size=256,
        rms_norm_eps=1e-5,
        num_attention_heads=64,
        qk_nope_head_dim=16,
        v_head_dim=16,
        kv_lora_rank=512,
        q_lora_rank=64,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=16,
        index_kpool=4,
    )
    is_kda = kind.startswith("kda")
    attention = KimiDeltaAttention if is_kda else NopeDSAttention
    reference = attention(config, hf_config).cuda().bfloat16()
    if kind == "kda_slow_decay":
        # Keep recurrent history significant across several CP intervals.
        reference.dt_bias.data.fill_(-5)
    parallel = copy.deepcopy(reference)
    parallel.cp_group = dist.group.WORLD
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    packed = SimpleNamespace(cu_seqlens_q=cu)
    full_input = torch.randn(total, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    start, end = rank * total // world_size, (rank + 1) * total // world_size
    local_input = full_input[start:end].detach().clone().requires_grad_()
    grad = torch.randn_like(full_input)
    # Later packed sequences cannot send any gradient into the first sequence.
    if len(lengths) > 1:
        grad[: lengths[0]] = 0
    kernel_grad = []
    if is_kda:
        # Capture dL/d(KDA output), before output normalization/projection.
        def capture_kernel_grad(module, inputs):
            inputs[0].register_hook(lambda value: kernel_grad.append(value.detach()))

        reference.o_norm.register_forward_pre_hook(capture_kernel_grad)
    expected, _ = reference(full_input, packed_seq_params=packed)
    actual, _ = parallel(local_input, packed_seq_params=packed)
    _assert_relative(actual, expected[start:end], f"{kind} output", tolerance=0.01)
    expected.backward(grad)
    actual.backward(grad[start:end])
    _assert_relative(local_input.grad, full_input.grad[start:end], f"{kind} input grad")
    gate_grads = _gate_grad_reference(reference, full_input, cu, kernel_grad[0]) if is_kda else {}
    for (name, param), (_, ref_param) in zip(parallel.named_parameters(), reference.named_parameters(), strict=True):
        if not param.requires_grad:
            continue
        assert param.grad is not None and ref_param.grad is not None, name
        reduced = param.grad.float()
        dist.all_reduce(reduced)
        if name in gate_grads:
            # Strong decay amplifies cancellation in FLA's BF16 backward:
            # CP=1 dt_bias itself differs from FP64 by ~9-12% here. CP changes
            # chunk boundaries, so require no accuracy regression against the
            # independent recurrence, allowing 0.5 percentage points of noise.
            oracle = gate_grads[name]
            baseline_error = (ref_param.grad.double() - oracle).norm() / oracle.norm().clamp_min(1e-8)
            _assert_relative(reduced, oracle, f"{kind} {name} FP64 grad", tolerance=baseline_error.item() + 0.005)
        else:
            _assert_relative(reduced, ref_param.grad, f"{kind} {name} grad")
    if len(lengths) > 1:
        local_prefix = max(0, min(end, lengths[0]) - start)
        assert torch.count_nonzero(local_input.grad[:local_prefix]) == 0
    if rank == 0:
        print(
            f"CP={world_size} {kind} lengths={lengths}: output, input and all parameter gradients passed", flush=True
        )


class _Embedding(torch.nn.Embedding):
    def forward(self, input_ids, position_ids=None):
        return super().forward(input_ids).transpose(0, 1)


class _LanguageModel(torch.nn.Module):
    def __init__(self, cp_group):
        super().__init__()
        self.embedding = _Embedding(40, 16)
        self.pg_collection = SimpleNamespace(cp=cp_group)

    def forward(self, *, decoder_input, **kwargs):
        return decoder_input


class _Vision(torch.nn.Linear):
    def forward(self, values, grid_thw):
        return SimpleNamespace(pooler_output=super().forward(values))


def _check_vision_injection(rank, world_size):
    torch.manual_seed(258)
    # Exercise the real wrapper's feature selection without allocating the
    # full GPT/vision tower; attention and vision kernels have separate tests.
    reference = Glm5NextModel.__new__(Glm5NextModel)
    torch.nn.Module.__init__(reference)
    reference.pre_process = True
    reference.image_token_id, reference.video_token_id = 35, 36
    reference.language_model = _LanguageModel(None)
    reference.model = torch.nn.Module()
    reference.model.visual = _Vision(3, 16)
    reference.cuda()
    parallel = copy.deepcopy(reference)
    parallel.language_model.pg_collection.cp = dist.group.WORLD
    tokens = torch.arange(256, device="cuda").remainder(32).unsqueeze(0)
    tokens[0, [1, 63, 64, 129, 200]] = 35
    tokens[0, [2, 126, 130, 254]] = 36
    images = torch.randn(5, 3, device="cuda", requires_grad=True)
    videos = torch.randn(4, 3, device="cuda", requires_grad=True)
    local_images, local_videos = [x.detach().clone().requires_grad_() for x in (images, videos)]
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 131, 256], dtype=torch.int32, device="cuda"))
    grids = torch.ones(1, 3, dtype=torch.int64, device="cuda")
    kwargs = dict(packed_seq_params=packed, image_grid_thw=grids, video_grid_thw=grids)
    start, end = rank * 256 // world_size, (rank + 1) * 256 // world_size
    expected = reference(tokens, pixel_values=images, pixel_values_videos=videos, **kwargs)
    actual = parallel(tokens[:, start:end], pixel_values=local_images, pixel_values_videos=local_videos, **kwargs)
    torch.testing.assert_close(actual, expected[start:end], rtol=0, atol=0)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad[start:end])
    for value, ref_value in ((local_images, images), (local_videos, videos)):
        dist.all_reduce(value.grad)
        torch.testing.assert_close(value.grad, ref_value.grad, rtol=1e-6, atol=1e-6)
    for param, ref_param in zip(parallel.parameters(), reference.parameters(), strict=True):
        dist.all_reduce(param.grad)
        torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-5, atol=1e-6)
    if rank == 0:
        print(f"CP={world_size} packed image/video feature selection and gradients passed", flush=True)


def _worker(rank, world_size, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(minutes=5)
    )
    try:
        for kind in ("kda", "kda_slow_decay", "mla"):
            for lengths in ([256], [131, 93, 32], [64, 64, 64, 64]):
                _check_attention(rank, world_size, kind, lengths)
        _check_vision_injection(rank, world_size)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_allgather_cp_matches_unsharded_attention_and_vision(cp_size):
    if torch.cuda.device_count() < cp_size:
        pytest.skip(f"Requires {cp_size} GPUs")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_worker, args=(cp_size, f"{directory}/rendezvous"), nprocs=cp_size, join=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
