import importlib
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from slime.backends.megatron_utils.alignment import deepgemm_moe_forward
from slime.backends.sglang_utils import compat as sglang_compat

NUM_GPUS = 1


def test_sglang_compat_prefers_current_module(monkeypatch):
    current_module = object()
    calls = []

    def fake_import(path):
        calls.append(path)
        return current_module

    monkeypatch.setattr(sglang_compat, "import_module", fake_import)

    assert sglang_compat.import_sglang_module("sglang.current", "sglang.legacy") is current_module
    assert calls == ["sglang.current"]


def test_sglang_compat_falls_back_for_moved_namespace(monkeypatch):
    legacy_module = object()
    calls = []

    def fake_import(path):
        calls.append(path)
        if path == "sglang.current.module":
            raise ModuleNotFoundError(name="sglang.current")
        return legacy_module

    monkeypatch.setattr(sglang_compat, "import_module", fake_import)

    assert sglang_compat.import_sglang_module("sglang.current.module", "sglang.legacy.module") is legacy_module
    assert calls == ["sglang.current.module", "sglang.legacy.module"]


def test_sglang_compat_does_not_mask_missing_dependency(monkeypatch):
    missing_dependency = ModuleNotFoundError(name="triton")

    def fake_import(path):
        raise missing_dependency

    monkeypatch.setattr(sglang_compat, "import_module", fake_import)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        sglang_compat.import_sglang_module("sglang.current.module", "sglang.legacy.module")

    assert exc_info.value is missing_dependency


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_configures_batch_invariant_in_megatron_actor(monkeypatch, value):
    state = {"enabled": False}
    deep_gemm = SimpleNamespace(get_batch_invariant=lambda: state["enabled"])
    wrapper = SimpleNamespace(configure_deep_gemm_batch_invariant=lambda enabled: state.update(enabled=enabled))
    monkeypatch.setenv("SGLANG_DEEPGEMM_BATCH_INVARIANT", value)

    assert deepgemm_moe_forward._configure_batch_invariant(deep_gemm, wrapper)
    assert state["enabled"]


def test_batch_invariant_actor_configuration_fails_closed(monkeypatch):
    deep_gemm = SimpleNamespace(get_batch_invariant=lambda: False)
    wrapper = SimpleNamespace(configure_deep_gemm_batch_invariant=lambda enabled: None)
    monkeypatch.setenv("SGLANG_DEEPGEMM_BATCH_INVARIANT", "1")

    with pytest.raises(RuntimeError, match="did not enable"):
        deepgemm_moe_forward._configure_batch_invariant(deep_gemm, wrapper)


def _small_bf16(shape, *, offset=0):
    values = torch.arange(offset, offset + torch.tensor(shape).prod().item(), dtype=torch.float32)
    values = ((values % 29) - 14) / 64
    return values.reshape(shape).to(torch.bfloat16)


def test_sorts_moe_chunks_into_preallocated_buffer():
    input_ = torch.tensor([[0], [1], [2], [3], [4], [5]])
    split_sizes = torch.tensor([2, 1, 3])
    sorted_idxs = torch.tensor([2, 0, 1])
    output = torch.empty_like(input_)

    actual = deepgemm_moe_forward._sort_chunks_into(
        input_,
        split_sizes,
        sorted_idxs,
        output,
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, torch.tensor([[3], [4], [5], [0], [1], [2]]))


def test_preallocated_combine_wrapper_avoids_original_allocation():
    class Dispatcher:
        tp_size = 1
        drop_and_pad = False
        num_local_experts = 3
        num_global_tokens_per_local_expert = torch.tensor([[2, 1, 0], [0, 1, 2]])
        restore_output_by_local_experts = torch.tensor([0, 3, 1, 4, 2, 5])

        def combine_preprocess(self, hidden_states):
            raise AssertionError("preallocated path must not call the allocating original")

    dispatcher = Dispatcher()
    assert deepgemm_moe_forward._wrap_preallocated_combine_preprocess(dispatcher)
    assert not deepgemm_moe_forward._wrap_preallocated_combine_preprocess(dispatcher)
    hidden_states = torch.arange(6).reshape(6, 1)
    combine_buffer = torch.empty_like(hidden_states)
    setattr(
        hidden_states,
        deepgemm_moe_forward._PREALLOCATED_COMBINE_BUFFER_ATTR,
        combine_buffer,
    )

    actual = dispatcher.combine_preprocess(hidden_states)

    assert actual.data_ptr() == combine_buffer.data_ptr()
    torch.testing.assert_close(actual, torch.tensor([[0], [1], [3], [2], [4], [5]]))


def test_combine_workspace_view_uses_caller_owned_bytes():
    module = torch.nn.Module()
    workspace = torch.empty(64, dtype=torch.uint8)
    setattr(module, deepgemm_moe_forward._COMBINE_WORKSPACE_ATTR, workspace)
    hidden_states = torch.empty((4, 8), dtype=torch.bfloat16)

    actual = deepgemm_moe_forward._combine_workspace_view(module, hidden_states)

    assert actual is not None
    assert actual.data_ptr() == workspace.data_ptr()
    assert actual.shape == hidden_states.shape
    assert actual.dtype == hidden_states.dtype


def test_combine_workspace_is_disabled_without_explicit_configuration(monkeypatch):
    monkeypatch.delenv("SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES", raising=False)

    assert deepgemm_moe_forward._combine_workspace_bytes() is None


def test_combine_workspace_uses_explicit_configuration(monkeypatch):
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES", "3758096384")

    assert deepgemm_moe_forward._combine_workspace_bytes() == 3758096384


def test_combine_workspace_view_rejects_undersized_storage():
    module = torch.nn.Module()
    setattr(
        module,
        deepgemm_moe_forward._COMBINE_WORKSPACE_ATTR,
        torch.empty(63, dtype=torch.uint8),
    )
    hidden_states = torch.empty((4, 8), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="too small"):
        deepgemm_moe_forward._combine_workspace_view(module, hidden_states)


def test_no_grad_dispatch_uses_workspace_and_preserves_alltoall_for_combine():
    class Dispatcher:
        tp_size = 1
        drop_and_pad = False
        num_local_experts = 3
        shared_experts = None
        config = SimpleNamespace(moe_permute_fusion=False)
        tokens_per_expert = torch.tensor([2, 2, 2])
        num_global_tokens_per_local_expert = torch.tensor([[2, 1, 0], [0, 1, 2]])
        sort_input_by_local_experts = torch.tensor([0, 3, 2, 5, 1, 4])

        @staticmethod
        def _maybe_dtoh_and_synchronize(_point, value):
            return value

        def dispatch_postprocess(self, *_args):
            raise AssertionError("no-grad workspace path must not call the allocating original")

    dispatcher = Dispatcher()
    workspace = torch.empty(64, dtype=torch.uint8)
    setattr(dispatcher, deepgemm_moe_forward._COMBINE_WORKSPACE_ATTR, workspace)
    assert deepgemm_moe_forward._wrap_preallocated_dispatch_postprocess(dispatcher)
    assert not deepgemm_moe_forward._wrap_preallocated_dispatch_postprocess(dispatcher)
    global_input = torch.arange(6).reshape(6, 1)
    global_probs = torch.arange(6, dtype=torch.float32)

    with torch.no_grad():
        hidden, counts, probs = dispatcher.dispatch_postprocess(
            global_input,
            global_probs,
        )

    assert hidden.data_ptr() == workspace.data_ptr()
    assert (
        getattr(
            hidden,
            deepgemm_moe_forward._PREALLOCATED_COMBINE_BUFFER_ATTR,
        ).data_ptr()
        == global_input.data_ptr()
    )
    torch.testing.assert_close(hidden, torch.tensor([[0], [1], [4], [5], [2], [3]]))
    torch.testing.assert_close(probs, torch.tensor([0.0, 1.0, 4.0, 5.0, 2.0, 3.0]))
    torch.testing.assert_close(counts, torch.tensor([2, 2, 2]))


def test_grad_dispatch_preserves_alltoall_for_combine():
    class Dispatcher:
        tp_size = 1
        drop_and_pad = False
        num_local_experts = 3
        shared_experts = None
        config = SimpleNamespace(moe_permute_fusion=False)

        @staticmethod
        def dispatch_postprocess(global_input, global_probs):
            return global_input.flip(0), torch.tensor([2, 2, 2]), global_probs.flip(0)

    dispatcher = Dispatcher()
    assert deepgemm_moe_forward._wrap_preallocated_dispatch_postprocess(dispatcher)
    global_input = torch.arange(6.0, requires_grad=True).reshape(6, 1)
    global_probs = torch.arange(6.0, requires_grad=True)

    hidden, counts, probs = dispatcher.dispatch_postprocess(global_input, global_probs)

    assert (
        getattr(
            hidden,
            deepgemm_moe_forward._PREALLOCATED_COMBINE_BUFFER_ATTR,
        ).data_ptr()
        == global_input.data_ptr()
    )
    torch.testing.assert_close(hidden, global_input.flip(0))
    torch.testing.assert_close(probs, global_probs.flip(0))
    torch.testing.assert_close(counts, torch.tensor([2, 2, 2]))


def test_preallocated_combine_wrapper_preserves_gradients():
    class Dispatcher:
        tp_size = 1
        drop_and_pad = False
        num_local_experts = 3
        num_global_tokens_per_local_expert = torch.tensor([[2, 1, 0], [0, 1, 2]])
        restore_output_by_local_experts = torch.tensor([0, 3, 1, 4, 2, 5])

        @staticmethod
        def combine_preprocess(_hidden_states):
            raise AssertionError("preallocated grad path must not call the allocating original")

    dispatcher = Dispatcher()
    assert deepgemm_moe_forward._wrap_preallocated_combine_preprocess(dispatcher)
    hidden_leaf = torch.arange(6.0, requires_grad=True)
    hidden_states = hidden_leaf.reshape(6, 1)
    alltoall_leaf = torch.arange(10.0, 16.0, requires_grad=True)
    combine_buffer = (alltoall_leaf * 1).reshape(6, 1)
    setattr(
        hidden_states,
        deepgemm_moe_forward._PREALLOCATED_COMBINE_BUFFER_ATTR,
        combine_buffer,
    )

    actual = dispatcher.combine_preprocess(hidden_states)
    weights = torch.arange(1.0, 7.0).reshape(6, 1)
    (actual * weights).sum().backward()

    assert actual.data_ptr() == combine_buffer.data_ptr()
    torch.testing.assert_close(actual, torch.tensor([[0.0], [1.0], [3.0], [2.0], [4.0], [5.0]]))
    torch.testing.assert_close(
        hidden_leaf.grad,
        torch.tensor([1.0, 2.0, 4.0, 3.0, 5.0, 6.0]),
    )
    torch.testing.assert_close(alltoall_leaf.grad, torch.zeros_like(alltoall_leaf))


def test_no_grad_token_combine_writes_alltoall_into_workspace(monkeypatch):
    class Group:
        @staticmethod
        def size():
            return 2

    class Dispatcher:
        ep_group = Group()
        input_splits = [3, 3]
        output_splits = [3, 3]

        def token_combine(self, *_args, **_kwargs):
            raise AssertionError("workspace path must not call the allocating original")

    dispatcher = Dispatcher()
    workspace = torch.empty(64, dtype=torch.uint8)
    setattr(dispatcher, deepgemm_moe_forward._COMBINE_WORKSPACE_ATTR, workspace)
    assert deepgemm_moe_forward._wrap_preallocated_token_combine(dispatcher)
    assert not deepgemm_moe_forward._wrap_preallocated_token_combine(dispatcher)
    hidden_states = torch.arange(6).reshape(6, 1)
    setattr(
        hidden_states,
        deepgemm_moe_forward._PREALLOCATED_TOKEN_COMBINE_ATTR,
        True,
    )
    calls = []

    def fake_all_to_all_single(output, input_, **kwargs):
        calls.append(kwargs)
        output.copy_(input_.flip(0))

    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all_single)
    with torch.no_grad():
        actual = dispatcher.token_combine(hidden_states)

    assert actual.data_ptr() == workspace.data_ptr()
    torch.testing.assert_close(actual, hidden_states.flip(0))
    assert calls == [
        {
            "output_split_sizes": [3, 3],
            "input_split_sizes": [3, 3],
            "group": dispatcher.ep_group,
        }
    ]


def test_router_probability_fp32_is_chunked_inplace(monkeypatch):
    monkeypatch.setattr(deepgemm_moe_forward, "_ROUTER_PROBABILITY_CHUNK_ROWS", 2)
    output = _small_bf16((5, 128), offset=17)
    original = output.clone()
    probabilities = torch.tensor([0.125, 0.25, 0.5, 1.25, 2.0], dtype=torch.bfloat16)
    expected = (original.float() * probabilities.float().reshape(-1, 1)).to(output.dtype)
    storage_pointer = output.data_ptr()

    actual = deepgemm_moe_forward._apply_router_probability_fp32_inplace(
        output,
        probabilities,
    )

    assert actual.data_ptr() == storage_pointer
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_router_probability_rejects_row_mismatch():
    with pytest.raises(RuntimeError, match="row mismatch"):
        deepgemm_moe_forward._apply_router_probability_fp32_inplace(
            torch.empty((3, 128), dtype=torch.bfloat16),
            torch.ones(2),
        )


def test_router_probability_grad_fp32_is_chunked_exact(monkeypatch):
    monkeypatch.setattr(deepgemm_moe_forward, "_ROUTER_PROBABILITY_CHUNK_ROWS", 2)
    grad_output = _small_bf16((5, 128), offset=29)
    down_output = _small_bf16((5, 128), offset=47)
    expected = (grad_output.float() * down_output.float()).sum(dim=-1, keepdim=True)

    actual = deepgemm_moe_forward._router_probability_grad_fp32_chunked(
        grad_output,
        down_output,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_valid_rows_are_compacted_inplace_in_chunks(monkeypatch):
    monkeypatch.setattr(deepgemm_moe_forward, "_UNPAD_CHUNK_ROWS", 2)
    padded = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    original = padded.clone()
    valid_rows = torch.tensor([0, 1, 4, 6, 7], dtype=torch.long)
    expected = original.index_select(0, valid_rows)
    storage_pointer = padded.data_ptr()

    actual = deepgemm_moe_forward._compact_valid_rows_inplace(
        padded,
        valid_rows,
    )

    assert actual.data_ptr() == storage_pointer
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deepep_scatter_preserves_topk_order_and_alignment_padding():
    hidden = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=torch.bfloat16,
    )
    topk_indices = torch.tensor(
        [[0, 1], [1, -1], [0, torch.iinfo(torch.int64).max]],
        dtype=torch.int64,
    )
    topk_weights = torch.tensor(
        [[0.1, 0.2], [0.3, 0.0], [0.4, 0.0]],
        dtype=torch.float32,
    )
    # DeepEP reports aligned counts; expert 0 has one explicit padding row.
    tokens_per_expert = torch.tensor([3, 2], dtype=torch.int32)

    permuted_hidden, permuted_probs, output_index, sanitized, routing_map, all_routes_valid = (
        deepgemm_moe_forward._scatter_deepep_routes_with_padding(
            hidden,
            topk_indices,
            topk_weights,
            tokens_per_expert,
        )
    )

    torch.testing.assert_close(
        permuted_hidden,
        torch.tensor(
            [[1.0, 10.0], [3.0, 30.0], [0.0, 0.0], [1.0, 10.0], [2.0, 20.0]],
            dtype=torch.bfloat16,
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        permuted_probs,
        torch.tensor([0.1, 0.4, 0.0, 0.2, 0.3]),
    )
    assert output_index.tolist() == [[0, 3], [4, -1], [1, -1]]
    assert not all_routes_valid
    assert sanitized.tolist() == [[0, 1], [1, -1], [0, -1]]
    assert routing_map.tolist() == [[True, True], [False, True], [True, False]]


def test_deepep_scatter_can_return_the_reusable_route_positions():
    hidden = _small_bf16((3, 16))
    topk_indices = torch.tensor([[0, 1], [1, -1], [0, -1]], dtype=torch.int64)
    topk_weights = torch.tensor(
        [[0.1, 0.2], [0.3, 0.0], [0.4, 0.0]],
        dtype=torch.float32,
    )

    *_, route_positions = deepgemm_moe_forward._scatter_deepep_routes_with_padding(
        hidden,
        topk_indices,
        topk_weights,
        torch.tensor([3, 2], dtype=torch.int32),
        return_route_positions=True,
    )

    torch.testing.assert_close(
        route_positions,
        torch.tensor([[0, 0], [0, 1], [1, 0], [2, 0]], dtype=torch.long),
    )


def test_deepep_scatter_exact_route_count_reuses_device_counts(monkeypatch):
    hidden = _small_bf16((3, 16))
    topk_indices = torch.tensor([[0, 1], [1, -1], [0, -1]], dtype=torch.int64)
    topk_weights = torch.tensor(
        [[0.1, 0.2], [0.3, 0.0], [0.4, 0.0]],
        dtype=torch.float32,
    )
    tokens_per_expert = torch.tensor([2, 2], dtype=torch.int32)
    original_to = torch.Tensor.to

    def reject_cpu_count_upload(tensor, *args, **kwargs):
        if tensor is tokens_per_expert:
            raise AssertionError("exact unpadded DeepEP counts must not be uploaded")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", reject_cpu_count_upload)
    permuted_hidden, permuted_probs, output_index, *_ = deepgemm_moe_forward._scatter_deepep_routes_with_padding(
        hidden,
        topk_indices,
        topk_weights,
        tokens_per_expert,
        expected_route_count=4,
    )

    torch.testing.assert_close(
        permuted_hidden,
        hidden.index_select(0, torch.tensor([0, 2, 0, 1])),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(permuted_probs, torch.tensor([0.1, 0.4, 0.2, 0.3]))
    assert output_index.tolist() == [[0, 2], [3, -1], [1, -1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_route_position_compaction_is_identical_to_nonzero():
    from slime.backends.megatron_utils.alignment.deterministic_route_kernels import compact_route_positions

    valid = torch.tensor(
        [[True, False, True], [False, True, True], [True, False, False]],
        device="cuda",
        dtype=torch.bool,
    )
    expected = torch.nonzero(valid, as_tuple=False)

    actual = compact_route_positions(valid, expected.shape[0])

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deepep_scatter_backward_does_not_retain_dispatch_buffer():
    hidden_leaf = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    hidden = hidden_leaf * 1
    dispatch_pointer = hidden.data_ptr()
    returned_grad_pointers = []
    hidden.register_hook(lambda grad: returned_grad_pointers.append(grad.data_ptr()))
    topk_indices = torch.tensor([[0, 1], [1, -1], [0, -1]], dtype=torch.int64)
    topk_weights = torch.tensor(
        [[0.1, 0.2], [0.3, 0.0], [0.4, 0.0]],
        dtype=torch.float32,
    )
    tokens_per_expert = torch.tensor([3, 2], dtype=torch.int32)

    permuted, _, _, _, _, _ = deepgemm_moe_forward._scatter_deepep_routes_with_padding(
        hidden,
        topk_indices,
        topk_weights,
        tokens_per_expert,
    )
    assert not hasattr(
        permuted,
        deepgemm_moe_forward._PREALLOCATED_COMBINE_BUFFER_ATTR,
    )

    grad = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=torch.bfloat16,
    )
    permuted.backward(grad)
    expected = torch.tensor(
        [[6.0, 8.0], [7.0, 8.0], [3.0, 4.0]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(hidden_leaf.grad, expected, rtol=0, atol=0)
    assert returned_grad_pointers[0] != dispatch_pointer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_deterministic_deepep_scatter_is_bitwise_forward_and_backward():
    torch.manual_seed(20260802)
    num_tokens, hidden_size, topk = 11, 384, 4
    hidden = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    valid_slots = num_tokens * topk - 5
    total_rows = valid_slots + 3
    valid_positions = torch.randperm(num_tokens * topk, device="cuda")[:valid_slots]
    output_index = torch.full(
        (num_tokens, topk),
        -1,
        device="cuda",
        dtype=torch.int64,
    )
    output_index.reshape(-1).index_copy_(
        0,
        valid_positions,
        torch.randperm(valid_slots, device="cuda", dtype=torch.int64),
    )

    actual = deepgemm_moe_forward._DeepEPScatterWithDeterministicBackward.apply(
        hidden,
        output_index,
        total_rows,
    )
    expected = torch.zeros_like(actual)
    positions = torch.nonzero(output_index >= 0, as_tuple=False)
    expected.index_copy_(
        0,
        output_index[positions[:, 0], positions[:, 1]],
        hidden.detach().index_select(0, positions[:, 0]),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected_grad = torch.zeros_like(hidden)
    for column in range(topk):
        route_rows = output_index[:, column]
        valid = route_rows >= 0
        selected = grad_output.index_select(0, route_rows.clamp(min=0))
        selected.masked_fill_(~valid.unsqueeze(1), 0)
        expected_grad.add_(selected)
    torch.testing.assert_close(hidden.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_fused_static_route_gradient_is_bitwise(monkeypatch):
    torch.manual_seed(20260803)
    num_tokens, hidden_size, topk = 13, 384, 4
    num_routes = num_tokens * topk
    original_routes = torch.randn(
        num_routes,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    actual_routes = original_routes.clone()
    weights = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
    output_index = torch.randperm(num_routes, device="cuda", dtype=torch.int64).reshape(
        num_tokens,
        topk,
    )
    grad_output = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )

    flat_route_rows = output_index.reshape(-1)
    token_rows = torch.arange(num_tokens, device="cuda").repeat_interleave(topk)
    token_grads = grad_output.index_select(0, token_rows)
    expected_routes = torch.empty_like(original_routes)
    expected_routes.index_copy_(
        0,
        flat_route_rows,
        (token_grads.float() * weights.reshape(-1, 1)).to(torch.bfloat16),
    )
    expected_weights = (
        (token_grads.float() * original_routes.index_select(0, flat_route_rows).float())
        .sum(dim=-1)
        .reshape_as(weights)
    )

    actual_weights = torch.empty_like(weights)
    deepgemm_moe_forward._ordered_route_backward(
        route_values=actual_routes,
        topk_weights=weights,
        output_index=output_index,
        grad_output=grad_output,
        grad_routes=actual_routes,
        grad_weights=actual_weights,
        static_mapping_valid=True,
    )

    torch.testing.assert_close(actual_routes, expected_routes, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def test_ordered_ep_gather_backward_matches_weighted_route_sum(monkeypatch):
    kernels = SimpleNamespace(ep_gather=None)
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "import_sglang_module",
        lambda current_path, legacy_path: kernels,
    )

    def fake_ep_gather(input_tensor, recv_ids, recv_weights, input_index, output):
        output.zero_()
        for token in range(recv_ids.shape[0]):
            accumulator = torch.zeros(input_tensor.shape[1], dtype=torch.float32)
            for column in range(recv_ids.shape[1]):
                row = int(input_index[token, column])
                if row >= 0:
                    accumulator += input_tensor[row].float() * recv_weights[token, column]
            output[token].copy_(accumulator.to(output.dtype))

    monkeypatch.setattr(kernels, "ep_gather", fake_ep_gather)
    hidden = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weights = torch.tensor(
        [[0.25, 0.75], [0.5, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    indices = torch.tensor([[0, 1], [0, -1]], dtype=torch.int64)
    output_index = torch.tensor([[0, 2], [1, -1]], dtype=torch.int64)
    grad = torch.tensor([[0.5, 1.0], [1.5, -0.5]], dtype=torch.bfloat16)

    output = deepgemm_moe_forward._SGLangEPGatherWithBF16Backward.apply(
        hidden,
        indices,
        weights,
        output_index,
        False,
        False,
    )
    output.backward(grad)

    hidden_ref = hidden.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()
    reference = torch.stack(
        [
            hidden_ref[0].float() * weights_ref[0, 0] + hidden_ref[2].float() * weights_ref[0, 1],
            hidden_ref[1].float() * weights_ref[1, 0],
        ]
    ).to(torch.bfloat16)
    reference.backward(grad)

    torch.testing.assert_close(output, reference.detach(), rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, weights_ref.grad, rtol=0, atol=0)


def test_deepep_route_handle_received_rows_supports_normal_layouts():
    intranode = (
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty((7,), dtype=torch.int32),
        torch.empty(1),
        torch.empty(1),
    )
    internode = (
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty((11, 2), dtype=torch.int32),
        torch.empty(1),
        torch.empty(1),
    )

    assert deepgemm_moe_forward._deepep_route_handle_received_rows(intranode) == 7
    assert deepgemm_moe_forward._deepep_route_handle_received_rows(internode) == 11


def test_deepep_alignment_is_disabled_outside_deterministic_mode(monkeypatch):
    def fail_if_model_is_inspected(_model):
        raise AssertionError("non-deterministic model must not be patched")

    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_normalize_model_chunks",
        fail_if_model_is_inspected,
    )
    args = SimpleNamespace(
        deterministic_mode=False,
        moe_enable_deepep=True,
        megatron_deepgemm_moe_forward_layers=(3,),
    )

    deepgemm_moe_forward.enable_sglang_deepep_moe_alignment(args, object(), "")


def test_deepep_alignment_captures_ordered_topk_without_r3(monkeypatch):
    from slime.utils import routing_replay

    monkeypatch.delenv("ENABLE_ROUTING_REPLAY", raising=False)
    monkeypatch.setattr(routing_replay, "ORDERED_TOPK_CAPTURE_ROUTER", None)

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        del num_groups, group_topk
        return torch.topk(scores, k=topk, dim=1)

    wrapped_compute_topk = routing_replay.get_routing_replay_compute_topk(compute_topk)

    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(moe_router_topk_scaling_factor=2.5)

        def routing(self, scores):
            return scores

        def forward(self, scores):
            selected_probs, selected_indices = wrapped_compute_topk(scores, 4)
            routing_map = torch.zeros_like(scores, dtype=torch.bool)
            routing_map.scatter_(1, selected_indices, True)
            dense_probs = torch.zeros_like(scores)
            dense_probs.scatter_(1, selected_indices, selected_probs)
            return routing_map, dense_probs, selected_indices

    class _DeepepManager:
        num_experts = 8
        capacity_factor = None

        def __init__(self):
            self.original_setup_calls = 0

        def setup_metadata(self, *_args):
            self.original_setup_calls += 1

        def dispatch(self, hidden_states, *_args, **_kwargs):
            return hidden_states

        def get_permuted_hidden_states_by_experts(self, hidden_states):
            return hidden_states

        def get_restored_hidden_states_by_experts(self, hidden_states):
            return hidden_states

    class Dispatcher:
        def __init__(self, manager):
            self._comm_manager = manager

        def token_combine(self, output):
            return output

        def combine_postprocess(self, output):
            return output

    router = Router()
    manager = _DeepepManager()
    mlp = SimpleNamespace(
        router=router,
        token_dispatcher=Dispatcher(manager),
        experts=SimpleNamespace(),
        config=SimpleNamespace(moe_latent_size=None),
        combine=lambda output: output,
    )
    assert deepgemm_moe_forward._patch_sglang_deepep_layer(mlp, global_layer=3)

    scores = torch.tensor(
        [
            [0.1, 0.9, 0.7, 0.8, 0.4, 0.5, 0.6, 0.2],
            [0.8, 0.2, 0.6, 0.1, 0.9, 0.7, 0.5, 0.3],
        ],
        dtype=torch.float32,
    )
    routing_map, dense_probs, expected_indices = router(scores)
    manager.setup_metadata(routing_map, dense_probs, None)

    torch.testing.assert_close(
        expected_indices,
        torch.topk(scores, k=4, dim=1, sorted=False).indices,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(manager.token_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(
        manager.token_probs,
        dense_probs.gather(1, expected_indices),
        rtol=0,
        atol=0,
    )
    assert manager.original_setup_calls == 0
    assert routing_replay.ORDERED_TOPK_CAPTURE_ROUTER is None
    assert not hasattr(router, "routing_replay")
    assert not hasattr(router, "_slime_ordered_topk_indices")

    # Megatron 1dcf0dafa splits combine/postprocess.  Its adapter must retain
    # SGLang's single shared + alpha * routed operation instead of rounding a
    # separately scaled BF16 routed tensor first.
    torch_add = torch.add
    add_calls = []

    def record_add(input, other, *, alpha=1):
        add_calls.append(alpha)
        return torch_add(input, other, alpha=alpha)

    monkeypatch.setattr(deepgemm_moe_forward.torch, "add", record_add)
    shared = torch.tensor([1.0], dtype=torch.bfloat16)
    routed = torch.tensor([2.0], dtype=torch.bfloat16)
    torch.testing.assert_close(
        mlp.postprocess(routed, shared),
        torch_add(shared, routed, alpha=2.5),
        rtol=0,
        atol=0,
    )
    assert add_calls == [2.5]


def test_topk_order_alignment_is_scoped_to_registered_router(monkeypatch):
    from slime.utils import routing_replay

    monkeypatch.delenv("ENABLE_ROUTING_REPLAY", raising=False)
    monkeypatch.setattr(routing_replay, "ORDERED_TOPK_CAPTURE_ROUTER", None)
    scores = torch.tensor(
        [[0.1, 0.9, 0.7, 0.8, 0.4, 0.5, 0.6, 0.2]],
        dtype=torch.float32,
    )

    def megatron_topk(values, topk, num_groups=None, group_topk=None):
        del num_groups, group_topk
        return torch.topk(values, k=topk, dim=1)

    wrapped = routing_replay.get_routing_replay_compute_topk(megatron_topk)
    _, actual = wrapped(scores, 4)

    expected = torch.topk(scores, k=4, dim=1, sorted=True).indices
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sglang_fused_sigmoid_topk_forward_and_backward(monkeypatch):
    from slime.utils import routing_replay

    router = SimpleNamespace()
    monkeypatch.setattr(routing_replay, "ORDERED_TOPK_CAPTURE_ROUTER", router)

    def fake_fused_topk(logits, expert_bias, topk):
        scores = torch.sigmoid(logits)
        indices = torch.topk(scores + expert_bias, topk, dim=1, sorted=False).indices
        selected = scores.gather(1, indices)
        return selected / selected.sum(dim=-1, keepdim=True), indices.to(torch.int32)

    monkeypatch.setattr(routing_replay, "_run_sglang_fused_sigmoid_topk", fake_fused_topk)
    logits = torch.tensor(
        [[-1.5, 0.25, 1.75, -0.5], [0.75, -2.0, 0.5, 1.25]],
        dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.tensor([0.1, -0.2, 0.05, 0.3], dtype=torch.float32)
    grad_weights = torch.tensor([[0.5, -1.25], [1.5, 0.75]], dtype=torch.float32)

    weights, indices = routing_replay.maybe_sglang_fused_sigmoid_topk(
        logits,
        2,
        use_pre_softmax=True,
        score_function="sigmoid",
        expert_bias=bias,
    )
    (weights * grad_weights).sum().backward()

    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_scores = torch.sigmoid(reference_logits)
    reference_selected = reference_scores.gather(1, indices.long())
    reference_weights = reference_selected / reference_selected.sum(dim=-1, keepdim=True)
    (reference_weights * grad_weights).sum().backward()

    torch.testing.assert_close(weights, reference_weights, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        routing_replay.consume_ordered_topk(router),
        indices,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sglang_fused_sigmoid_topk_is_bitwise_batch_invariant():
    try:
        moe_fused_gate = sglang_compat.import_sglang_module(
            "sglang.kernels.ops.moe.moe_fused_gate",
            "sglang.jit_kernel.moe_fused_gate",
        ).moe_fused_gate
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"SGLang fused router is unavailable: {exc}")

    generator = torch.Generator(device="cuda").manual_seed(1234)
    logits = torch.randn((73, 256), device="cuda", dtype=torch.float32, generator=generator)
    bias = torch.randn((256,), device="cuda", dtype=torch.float32, generator=generator) * 0.125
    expected_weights, expected_indices = moe_fused_gate(
        logits,
        bias,
        8,
        scoring_func="sigmoid",
        renormalize=True,
    )

    for chunk_size in (1, 3, 17, 64):
        actual_weights = torch.empty_like(expected_weights)
        actual_indices = torch.empty_like(expected_indices)
        for start in range(0, logits.shape[0], chunk_size):
            stop = min(start + chunk_size, logits.shape[0])
            actual_weights[start:stop], actual_indices[start:stop] = moe_fused_gate(
                logits[start:stop],
                bias,
                8,
                scoring_func="sigmoid",
                renormalize=True,
            )
        torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def test_static_combine_backward_matches_dynamic_path():
    route_values = _small_bf16((8, 128), offset=3)
    weights = torch.tensor(
        [[0.25, 0.75], [0.5, 1.0], [1.25, 0.125], [0.875, 0.625]],
        dtype=torch.float32,
    )
    output_index = torch.tensor(
        [[2, 0], [7, 3], [1, 6], [5, 4]],
        dtype=torch.int64,
    )
    grad_output = _small_bf16((4, 128), offset=19)

    dynamic_routes = torch.zeros_like(route_values)
    dynamic_weights = torch.zeros_like(weights)
    deepgemm_moe_forward._ordered_route_backward(
        route_values=route_values,
        topk_weights=weights,
        output_index=output_index,
        grad_output=grad_output,
        grad_routes=dynamic_routes,
        grad_weights=dynamic_weights,
    )

    static_routes = torch.zeros_like(route_values)
    static_weights = torch.zeros_like(weights)
    deepgemm_moe_forward._ordered_route_backward(
        route_values=route_values,
        topk_weights=weights,
        output_index=output_index,
        grad_output=grad_output,
        grad_routes=static_routes,
        grad_weights=static_weights,
        static_mapping_valid=True,
    )

    torch.testing.assert_close(static_routes, dynamic_routes, rtol=0, atol=0)
    torch.testing.assert_close(static_weights, dynamic_weights, rtol=0, atol=0)


def test_static_combine_backward_can_reuse_route_value_storage():
    original_routes = _small_bf16((8, 128), offset=3)
    route_values = original_routes.clone()
    weights = torch.tensor(
        [[0.25, 0.75], [0.5, 1.0], [1.25, 0.125], [0.875, 0.625]],
        dtype=torch.float32,
    )
    output_index = torch.tensor(
        [[2, 0], [7, 3], [1, 6], [5, 4]],
        dtype=torch.int64,
    )
    grad_output = _small_bf16((4, 128), offset=19)

    expected_routes = torch.zeros_like(route_values)
    expected_weights = torch.zeros_like(weights)
    deepgemm_moe_forward._ordered_route_backward(
        route_values=original_routes,
        topk_weights=weights,
        output_index=output_index,
        grad_output=grad_output,
        grad_routes=expected_routes,
        grad_weights=expected_weights,
    )

    actual_weights = torch.zeros_like(weights)
    storage_ptr = route_values.untyped_storage().data_ptr()
    deepgemm_moe_forward._ordered_route_backward(
        route_values=route_values,
        topk_weights=weights,
        output_index=output_index,
        grad_output=grad_output,
        grad_routes=route_values,
        grad_weights=actual_weights,
        static_mapping_valid=True,
    )

    assert route_values.untyped_storage().data_ptr() == storage_ptr
    torch.testing.assert_close(route_values, expected_routes, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def test_static_combine_backward_handles_padding_with_reused_storage():
    original_routes = _small_bf16((6, 128), offset=7)
    route_values = original_routes.clone()
    weights = torch.tensor(
        [[0.25, 0.75], [0.5, 0.0], [1.25, 0.125]],
        dtype=torch.float32,
    )
    # Route row 5 is expert-alignment padding and the second slot of token 1
    # is a masked data-padding route.
    output_index = torch.tensor(
        [[2, 0], [4, -1], [1, 3]],
        dtype=torch.int64,
    )
    grad_output = _small_bf16((3, 128), offset=23)

    expected_routes = torch.zeros_like(original_routes)
    expected_weights = torch.zeros_like(weights)
    deepgemm_moe_forward._ordered_route_backward(
        route_values=original_routes,
        topk_weights=weights,
        output_index=output_index,
        grad_output=grad_output,
        grad_routes=expected_routes,
        grad_weights=expected_weights,
    )

    actual_weights = torch.zeros_like(weights)
    storage_ptr = route_values.untyped_storage().data_ptr()
    deepgemm_moe_forward._ordered_route_backward(
        route_values=route_values,
        topk_weights=weights,
        output_index=output_index,
        grad_output=grad_output,
        grad_routes=route_values,
        grad_weights=actual_weights,
        static_mapping_valid=False,
    )

    assert route_values.untyped_storage().data_ptr() == storage_ptr
    torch.testing.assert_close(route_values, expected_routes, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def test_low_latency_ep_gather_preserves_order_and_backward(monkeypatch):
    kernels = SimpleNamespace(ep_gather=None)
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "import_sglang_module",
        lambda current_path, legacy_path: kernels,
    )

    calls = []

    def fake_ep_gather(hidden_states, topk_ids, topk_weights, output_index, output):
        calls.append(
            (
                topk_ids.clone(),
                topk_weights.clone(),
                output_index.clone(),
                output.dtype,
            )
        )
        for token in range(topk_ids.shape[0]):
            accumulator = torch.zeros(hidden_states.shape[1], dtype=torch.float32)
            for slot in range(topk_ids.shape[1]):
                route = int(output_index[token, slot])
                if route >= 0:
                    accumulator.add_(hidden_states[route].float() * topk_weights[token, slot])
            output[token].copy_(accumulator.to(output.dtype))

    monkeypatch.setattr(kernels, "ep_gather", fake_ep_gather)
    routes = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    ordered = torch.tensor([[2, 0], [1, -1]], dtype=torch.int64)
    weights = torch.tensor(
        [[0.75, 0.25], [0.5, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    output_index = torch.tensor([[2, 0], [1, -1]], dtype=torch.int64)
    grad = torch.tensor([[0.5, 1.0], [1.5, -0.5]], dtype=torch.bfloat16)

    output = deepgemm_moe_forward._SGLangEPGatherWithBF16Backward.apply(
        routes,
        ordered,
        weights,
        output_index,
        False,
        False,
    )
    output.backward(grad)

    expected = torch.stack(
        [
            routes.detach()[2].float() * weights.detach()[0, 0] + routes.detach()[0].float() * weights.detach()[0, 1],
            routes.detach()[1].float() * weights.detach()[1, 0],
        ]
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], ordered, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1], weights.detach(), rtol=0, atol=0)
    torch.testing.assert_close(calls[0][2], output_index, rtol=0, atol=0)
    assert calls[0][3] == torch.bfloat16

    expected_route_grad = torch.zeros_like(routes)
    expected_route_grad[2] = (grad[0].float() * weights.detach()[0, 0]).to(torch.bfloat16)
    expected_route_grad[0] = (grad[0].float() * weights.detach()[0, 1]).to(torch.bfloat16)
    expected_route_grad[1] = (grad[1].float() * weights.detach()[1, 0]).to(torch.bfloat16)
    torch.testing.assert_close(routes.grad, expected_route_grad, rtol=0, atol=0)


class TEColumnParallelGroupedLinear(torch.nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.num_gemms = num_experts
        self.use_bias = False
        for expert_index in range(num_experts):
            self.register_parameter(
                f"weight{expert_index}",
                torch.nn.Parameter(
                    _small_bf16(
                        (2 * ffn_hidden_size, hidden_size),
                        offset=expert_index * 7,
                    )
                ),
            )


class TERowParallelGroupedLinear(torch.nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.num_gemms = num_experts
        self.use_bias = False
        for expert_index in range(num_experts):
            self.register_parameter(
                f"weight{expert_index}",
                torch.nn.Parameter(
                    _small_bf16(
                        (hidden_size, ffn_hidden_size),
                        offset=expert_index * 11 + 3,
                    )
                ),
            )


class TEGroupedMLP(torch.nn.Module):
    """CPU fake with Megatron's probability-before-fc2 autograd graph."""

    def __init__(
        self,
        *,
        num_experts: int = 2,
        hidden_size: int = 128,
        ffn_hidden_size: int = 128,
    ):
        super().__init__()
        self.num_local_experts = num_experts
        self.config = SimpleNamespace(
            add_bias_linear=False,
            gated_linear_unit=True,
            moe_apply_probs_on_input=False,
            fp8=False,
            qat=False,
            swiglu_clamp_limit=None,
            hidden_size=hidden_size,
            moe_ffn_hidden_size=ffn_hidden_size,
        )
        self.activation_func = F.silu
        self.linear_fc1 = TEColumnParallelGroupedLinear(
            num_experts,
            hidden_size,
            ffn_hidden_size,
        )
        self.linear_fc2 = TERowParallelGroupedLinear(
            num_experts,
            hidden_size,
            ffn_hidden_size,
        )
        self.output_bias = object()
        self.forward_calls = 0

    def forward(self, hidden_states, tokens_per_expert, permuted_probs):
        self.forward_calls += 1
        counts = tuple(int(value) for value in tokens_per_expert.tolist())
        hidden_chunks = torch.split(hidden_states, counts)
        probability_chunks = torch.split(permuted_probs.reshape(-1, 1), counts)
        outputs = []
        for expert_index, (hidden, probability) in enumerate(zip(hidden_chunks, probability_chunks, strict=True)):
            fc1_weight = getattr(self.linear_fc1, f"weight{expert_index}")
            gate_up = F.linear(hidden, fc1_weight)
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            activated = (activated.float() * probability.float()).to(hidden.dtype)
            fc2_weight = getattr(self.linear_fc2, f"weight{expert_index}")
            outputs.append(F.linear(activated, fc2_weight))
        return torch.cat(outputs, dim=0), self.output_bias


def _post_fc2_probability_reference(module, hidden_states, tokens_per_expert, permuted_probs):
    counts = tuple(int(value) for value in tokens_per_expert.tolist())
    hidden_chunks = torch.split(hidden_states, counts)
    probability_chunks = torch.split(permuted_probs.reshape(-1, 1), counts)
    outputs = []
    for expert_index, (hidden, probability) in enumerate(zip(hidden_chunks, probability_chunks, strict=True)):
        fc1_weight = getattr(module.linear_fc1, f"weight{expert_index}")
        gate_up = F.linear(hidden, fc1_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (F.silu(gate.float()) * up.float()).to(hidden.dtype)
        fc2_weight = getattr(module.linear_fc2, f"weight{expert_index}")
        down_output = F.linear(activated, fc2_weight)
        outputs.append((down_output.float() * probability.float()).to(hidden.dtype))
    return torch.cat(outputs, dim=0)


class FakeDeepGEMMOps:
    def __init__(self):
        self.weight_inputs = []
        self.activation_inputs = []
        self.activation_input_pointers = []
        self.m_indices = []
        self.grouped_weight_shapes = []
        self.grouped_output_pointers = []
        self.raw_down_output = None

    def quantize_weight(self, weight, block_size):
        assert block_size == (128, 128)
        self.weight_inputs.append(weight.detach().clone())
        scale_shape = (
            (weight.shape[0] + 127) // 128,
            (weight.shape[1] + 127) // 128,
        )
        return (
            weight.detach().clone(),
            torch.ones(scale_shape, dtype=torch.float32),
        )

    def quantize_activation(
        self,
        value,
        group_size,
        *,
        column_major_scales,
        scale_tma_aligned,
        scale_ue8m0,
    ):
        assert group_size == 128
        assert column_major_scales is False
        assert scale_tma_aligned is False
        assert scale_ue8m0 is False
        self.activation_input_pointers.append(value.data_ptr())
        self.activation_inputs.append(value.detach().clone())
        return (
            value.detach().clone(),
            torch.ones(
                (value.shape[0], value.shape[1] // group_size),
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def align_input_scale(scale):
        return scale

    def grouped_gemm(self, lhs, rhs, out, m_indices):
        self.m_indices.append(m_indices.detach().clone())
        self.grouped_output_pointers.append(out.data_ptr())
        inputs = lhs[0].float()
        weights = rhs[0].float()
        self.grouped_weight_shapes.append(tuple(weights.shape))
        for expert_index in range(weights.shape[0]):
            row_mask = m_indices == expert_index
            if row_mask.any():
                result = inputs[row_mask] @ weights[expert_index].transpose(0, 1)
                out[row_mask] = result.to(out.dtype)
        if out.shape[1] == 128:
            self.raw_down_output = out.detach().clone()

    @staticmethod
    def silu_and_mul(gate_up, out):
        gate, up = gate_up.chunk(2, dim=-1)
        out.copy_((F.silu(gate.float()) * up.float()).to(out.dtype))

    def as_ops(self):
        return deepgemm_moe_forward._DeepGEMMOps(
            quantize_weight=self.quantize_weight,
            quantize_activation=self.quantize_activation,
            align_input_scale=self.align_input_scale,
            grouped_gemm=self.grouped_gemm,
            silu_and_mul=self.silu_and_mul,
        )


@pytest.fixture
def parallelism_one(monkeypatch):
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_expert_tensor_parallel_world_size",
        lambda: 1,
    )


def test_deepgemm_forward_skips_te_and_differentiates_post_fc2_probability(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    monkeypatch.setattr(deepgemm_moe_forward, "_BACKWARD_CHUNK_ROWS", 1)
    module = TEGroupedMLP()
    counts = torch.tensor([1, 2], dtype=torch.int32)
    expected_input = _small_bf16((3, 128), offset=5).requires_grad_()
    expected_probs = torch.tensor([0.25, 1.5, 2.0], dtype=torch.float32, requires_grad=True)
    fixed_grad = _small_bf16((3, 128), offset=13)
    parameters = tuple(module.parameters())
    expected_output = _post_fc2_probability_reference(module, expected_input, counts, expected_probs)
    expected_gradients = torch.autograd.grad(
        expected_output,
        (expected_input, expected_probs, *parameters),
        fixed_grad,
    )
    module.forward_calls = 0

    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    assert deepgemm_moe_forward._wrap_te_grouped_mlp(
        module,
        "decoder.layers.0.mlp.experts",
    )
    backward_workspace = torch.empty(
        expected_input.numel() * expected_input.element_size(),
        dtype=torch.uint8,
    )
    setattr(
        module,
        deepgemm_moe_forward._COMBINE_WORKSPACE_ATTR,
        backward_workspace,
    )

    input_ = expected_input.detach().clone().requires_grad_()
    input_grad_pointers = []
    input_.register_hook(lambda grad: input_grad_pointers.append(grad.data_ptr()))
    probs = expected_probs.detach().clone().requires_grad_()
    output, bias = module(input_, counts, probs)

    assert bias is module.output_bias
    assert module.forward_calls == 0
    torch.testing.assert_close(output, expected_output.detach(), rtol=0, atol=0)

    output.backward(fixed_grad)
    actual_gradients = (input_.grad, probs.grad, *(parameter.grad for parameter in parameters))
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.06, atol=0.006)
    assert input_grad_pointers == [backward_workspace.data_ptr()]


def test_no_grad_forward_reuses_dispatched_input_without_changing_values(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    module = TEGroupedMLP()
    counts = torch.tensor([1, 2], dtype=torch.int32)
    hidden = _small_bf16((3, 128), offset=5)
    probs = torch.tensor([0.25, 1.5, 2.0], dtype=torch.float32)
    expected = _post_fc2_probability_reference(module, hidden.clone(), counts, probs)
    input_pointer = hidden.data_ptr()
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    assert deepgemm_moe_forward._wrap_te_grouped_mlp(
        module,
        "decoder.layers.0.mlp.experts",
    )

    with torch.no_grad():
        output, bias = module(hidden, counts, probs)

    assert bias is module.output_bias
    assert output.data_ptr() == input_pointer
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_no_grad_aligned_glm_layout_reuses_input_for_all_activation_scratch(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "2")
    module = TEGroupedMLP(num_experts=2, hidden_size=384, ffn_hidden_size=128)
    counts = torch.tensor([128, 128], dtype=torch.int32)
    hidden = _small_bf16((256, 384), offset=5)
    probs = torch.linspace(0.25, 1.25, 256, dtype=torch.float32)
    expected = _post_fc2_probability_reference(module, hidden.clone(), counts, probs)
    input_pointer = hidden.data_ptr()
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )

    output = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden,
        counts,
        probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
        reuse_input_buffer=True,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.data_ptr() == input_pointer
    # FC1 gate/up and FC2 output both begin at the consumed input storage.
    assert fake_ops.grouped_output_pointers == [input_pointer, input_pointer]
    gate_up_bytes = 256 * 2 * 128 * hidden.element_size()
    assert fake_ops.activation_input_pointers == [input_pointer, input_pointer + gate_up_bytes]


def test_no_grad_padded_glm_layout_reuses_padding_for_activation_scratch(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "2")
    module = TEGroupedMLP(num_experts=2, hidden_size=384, ffn_hidden_size=128)
    counts = torch.tensor([129, 130], dtype=torch.int32)
    hidden = _small_bf16((259, 384), offset=7)
    probs = torch.linspace(0.25, 1.25, 259, dtype=torch.float32)
    expected = _post_fc2_probability_reference(module, hidden.clone(), counts, probs)
    input_pointer = hidden.data_ptr()
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )

    output = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden,
        counts,
        probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
        reuse_input_buffer=True,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.data_ptr() == input_pointer
    padded_pointer = fake_ops.activation_input_pointers[0]
    assert padded_pointer != input_pointer
    assert fake_ops.grouped_output_pointers == [padded_pointer, padded_pointer]
    gate_up_bytes = 512 * 2 * 128 * hidden.element_size()
    assert fake_ops.activation_input_pointers == [
        padded_pointer,
        padded_pointer + gate_up_bytes,
    ]


def test_deepgemm_backward_skips_frozen_expert_weight_gradients(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    module = TEGroupedMLP()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    counts = torch.tensor([1, 2], dtype=torch.int32)
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_deepgemm_bf16_gemm_tn",
        lambda *_args, **_kwargs: pytest.fail("frozen expert weights must not compute wgrad"),
    )
    assert deepgemm_moe_forward._wrap_te_grouped_mlp(
        module,
        "decoder.layers.0.mlp.experts",
    )

    input_ = _small_bf16((3, 128), offset=5).requires_grad_()
    probs = torch.tensor(
        [0.25, 1.5, 2.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    output, _ = module(input_, counts, probs)
    output.backward(_small_bf16((3, 128), offset=13))

    assert input_.grad is not None
    assert probs.grad is not None
    assert all(parameter.grad is None for parameter in module.parameters())


def test_weight_order_layout_and_zero_count_expert_reach_grouped_gemm(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    # This assertion checks the all-experts weight ordering.  Keep its scope
    # independent of the benchmark's operation-local grouping override; the
    # grouped path itself is covered by the following workspace-bound test.
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "3")
    module = TEGroupedMLP(num_experts=3)
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    deepgemm_moe_forward._wrap_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )

    hidden = _small_bf16((3, 128), offset=2)
    counts = torch.tensor([2, 0, 1], dtype=torch.int64)
    probs = torch.tensor([0.5, 0.75, 1.25], dtype=torch.float32)
    output, _ = module(hidden, counts, probs)

    assert output.shape == hidden.shape
    assert len(fake_ops.weight_inputs) == 6
    for expert_index in range(3):
        torch.testing.assert_close(
            fake_ops.weight_inputs[expert_index],
            getattr(module.linear_fc1, f"weight{expert_index}"),
        )
        torch.testing.assert_close(
            fake_ops.weight_inputs[3 + expert_index],
            getattr(module.linear_fc2, f"weight{expert_index}"),
        )
    assert len(fake_ops.m_indices) == 2
    for m_indices in fake_ops.m_indices:
        assert m_indices.dtype == torch.int32
        torch.testing.assert_close(
            m_indices,
            torch.cat(
                (
                    torch.zeros(128, dtype=torch.int32),
                    torch.full((128,), 2, dtype=torch.int32),
                )
            ),
        )


def test_deterministic_forward_limits_peak_workspace_to_expert_groups(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    monkeypatch.setenv("SGLANG_DEEPGEMM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "1")
    module = TEGroupedMLP(num_experts=3)
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )
    counts = torch.tensor([2, 1, 3], dtype=torch.int64)
    hidden = _small_bf16((6, 128), offset=2)
    probs = torch.tensor([0.5, 0.75, 1.25, 0.25, 1.5, 2.0], dtype=torch.float32)

    output = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden,
        counts,
        probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
    )

    expected = _post_fc2_probability_reference(module, hidden, counts, probs)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert len(fake_ops.grouped_weight_shapes) == 6
    assert all(shape[0] == 1 for shape in fake_ops.grouped_weight_shapes)
    assert max(value.shape[0] for value in fake_ops.activation_inputs) == 128
    assert all(indices.unique().tolist() == [0] for indices in fake_ops.m_indices)


def test_grouped_weight_quantization_uses_dead_output_workspace(
    parallelism_one,
):
    del parallelism_one
    module = TEGroupedMLP(
        num_experts=2,
        hidden_size=384,
        ffn_hidden_size=128,
    )
    fake_ops = FakeDeepGEMMOps()
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )
    grouped_shape = (2, *layout.fc1_weight_shape)
    workspace = torch.empty(math.prod(grouped_shape), dtype=torch.bfloat16)

    grouped_qweight, grouped_scale = deepgemm_moe_forward._quantize_grouped_weights(
        module.linear_fc1,
        expected_shape=layout.fc1_weight_shape,
        layout=layout,
        input_device=workspace.device,
        module_name="decoder.layers.3.mlp.experts.linear_fc1",
        ops=fake_ops.as_ops(),
        grouped_qweight_workspace=workspace,
    )

    assert grouped_qweight.shape == grouped_shape
    assert grouped_qweight.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    assert grouped_scale.shape[0] == 2
    for expert_index in range(2):
        torch.testing.assert_close(
            grouped_qweight[expert_index],
            getattr(module.linear_fc1, f"weight{expert_index}"),
        )


def test_checkpoint_recompute_reuses_shared_workspace_for_grouped_outputs(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "1")
    module = TEGroupedMLP(num_experts=2)
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )
    workspace = torch.empty(128 * 2 * 128 * 2, dtype=torch.uint8)
    setattr(module, deepgemm_moe_forward._COMBINE_WORKSPACE_ATTR, workspace)
    counts = torch.tensor([2, 3], dtype=torch.int64)
    hidden = _small_bf16((5, 128), offset=2)
    probs = torch.tensor([0.5, 0.75, 1.25, 0.25, 1.5], dtype=torch.float32)
    expected = _post_fc2_probability_reference(module, hidden, counts, probs)

    output = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden,
        counts,
        probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
        reuse_input_buffer=False,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert fake_ops.grouped_output_pointers == [workspace.data_ptr()] * 4


def test_checkpoint_recompute_writes_group_outputs_into_final_storage(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "1")
    monkeypatch.delenv("SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES", raising=False)
    module = TEGroupedMLP(num_experts=2)
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )
    counts = torch.tensor([2, 3], dtype=torch.int64)
    hidden = _small_bf16((5, 128), offset=2)
    probs = torch.tensor([0.5, 0.75, 1.25, 0.25, 1.5], dtype=torch.float32)
    expected = _post_fc2_probability_reference(module, hidden, counts, probs)

    output = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden,
        counts,
        probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
        reuse_input_buffer=False,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    # FC2 for expert 0 writes at the start of the returned output's backing
    # storage; expert 1 writes after expert 0's two compact rows.  FC1 pointers
    # in the interleaved list still refer to their independent gate/up buffers.
    assert fake_ops.grouped_output_pointers[1] == output.data_ptr()
    assert fake_ops.grouped_output_pointers[3] == output[2:].data_ptr()
    # Each padded expert input is quantized from that same not-yet-written FC2
    # destination, avoiding a separate group-sized BF16 padding allocation.
    assert fake_ops.activation_input_pointers[0] == fake_ops.grouped_output_pointers[1]
    assert fake_ops.activation_input_pointers[2] == fake_ops.grouped_output_pointers[3]


def test_checkpoint_recompute_reuses_wide_final_output_for_fc1_and_activation_scratch(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "1")
    monkeypatch.delenv("SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES", raising=False)
    # GLM has hidden_size == 3 * ffn_hidden_size, so the final-output
    # destination can hold [gate, up, silu(gate) * up] without overlap.
    module = TEGroupedMLP(num_experts=2, hidden_size=384, ffn_hidden_size=128)
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )
    counts = torch.tensor([2, 3], dtype=torch.int64)
    hidden = _small_bf16((5, 384), offset=2)
    probs = torch.tensor([0.5, 0.75, 1.25, 0.25, 1.5], dtype=torch.float32)
    expected = _post_fc2_probability_reference(module, hidden, counts, probs)

    output = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden,
        counts,
        probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
        reuse_input_buffer=False,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    # FC1 gate/up and FC2 output both use the start of the current final-output
    # slice.  The quantized down input reads the disjoint tail after gate/up.
    assert fake_ops.grouped_output_pointers[0] == fake_ops.grouped_output_pointers[1]
    assert fake_ops.grouped_output_pointers[2] == fake_ops.grouped_output_pointers[3]
    gate_up_bytes = 128 * 2 * 128 * torch.empty((), dtype=torch.bfloat16).element_size()
    assert fake_ops.activation_input_pointers[1] == fake_ops.grouped_output_pointers[0] + gate_up_bytes
    assert fake_ops.activation_input_pointers[3] == fake_ops.grouped_output_pointers[2] + gate_up_bytes


def test_initial_no_grad_forward_does_not_overwrite_dispatch_workspace(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "1")
    module = TEGroupedMLP(num_experts=2)
    fake_ops = FakeDeepGEMMOps()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        fake_ops.as_ops,
    )
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )
    workspace = torch.empty(128 * 2 * 128 * 2, dtype=torch.uint8)
    setattr(module, deepgemm_moe_forward._COMBINE_WORKSPACE_ATTR, workspace)
    counts = torch.tensor([2, 3], dtype=torch.int64)
    hidden = _small_bf16((5, 128), offset=2)
    probs = torch.tensor([0.5, 0.75, 1.25, 0.25, 1.5], dtype=torch.float32)
    expected = _post_fc2_probability_reference(module, hidden, counts, probs)

    output = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden,
        counts,
        probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
        reuse_input_buffer=True,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert all(pointer != workspace.data_ptr() for pointer in fake_ops.grouped_output_pointers)


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_rejects_invalid_experts_per_forward_group(monkeypatch, value):
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", value)
    with pytest.raises(RuntimeError, match="must be a positive integer"):
        deepgemm_moe_forward._experts_per_forward_group(8)


def test_all_zero_rows_short_circuit_without_loading_cuda_ops(
    monkeypatch,
    parallelism_one,
):
    del parallelism_one
    module = TEGroupedMLP()
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        lambda: pytest.fail("zero-row path must not load or launch CUDA ops"),
    )
    deepgemm_moe_forward._wrap_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )

    output, bias = module(
        torch.empty((0, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.empty((0,), dtype=torch.float32),
    )

    assert output.shape == (0, 128)
    assert output.dtype == torch.bfloat16
    assert bias is module.output_bias


@pytest.mark.parametrize(
    ("counts", "probs", "message"),
    [
        (torch.tensor([1, 0]), torch.ones(2), "sum does not match"),
        (torch.tensor([3, -1]), torch.ones(2), "negative count"),
        (torch.tensor([1, 1]), torch.ones(3, 1), "one scalar per"),
    ],
)
def test_invalid_routing_metadata_fails_before_cuda_ops(
    monkeypatch,
    parallelism_one,
    counts,
    probs,
    message,
):
    del parallelism_one
    module = TEGroupedMLP()
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "_load_deepgemm_ops",
        lambda: pytest.fail("invalid metadata must fail before loading CUDA ops"),
    )

    with pytest.raises(RuntimeError, match=message):
        deepgemm_moe_forward._deepgemm_grouped_moe_forward(
            module,
            _small_bf16((2, 128)),
            counts,
            probs,
            layout=layout,
            module_name="decoder.layers.3.mlp.experts",
        )


def test_rejects_tp_or_expert_tp_greater_than_one(monkeypatch):
    module = TEGroupedMLP()
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_expert_tensor_parallel_world_size",
        lambda: 1,
    )
    with pytest.raises(RuntimeError, match="tensor model parallel size 1"):
        deepgemm_moe_forward._wrap_te_grouped_mlp(
            module,
            "decoder.layers.3.mlp.experts",
        )

    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_expert_tensor_parallel_world_size",
        lambda: 2,
    )
    with pytest.raises(RuntimeError, match="expert tensor parallel size 1"):
        deepgemm_moe_forward._wrap_te_grouped_mlp(
            module,
            "decoder.layers.3.mlp.experts",
        )


def test_rejects_transposed_or_individually_wrapped_expert_weights(parallelism_one):
    del parallelism_one
    bad_layout = TEGroupedMLP()
    bad_layout.linear_fc1.weight0 = torch.nn.Parameter(torch.empty((128, 256), dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match=r"expected \(256, 128\)"):
        deepgemm_moe_forward._wrap_te_grouped_mlp(
            bad_layout,
            "decoder.layers.3.mlp.experts",
        )

    child_wrapped = TEGroupedMLP()
    child_wrapped.linear_fc2._slime_deepgemm_forward_wrapped = True
    with pytest.raises(RuntimeError, match="individually wrapped"):
        deepgemm_moe_forward._wrap_te_grouped_mlp(
            child_wrapped,
            "decoder.layers.3.mlp.experts",
        )


def test_installer_uses_global_layer_number_with_wrapper_prefixes(parallelism_one):
    del parallelism_one

    class Layer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number
            self.mlp = torch.nn.Module()
            self.mlp.experts = TEGroupedMLP(num_experts=1)

    gpt = torch.nn.Module()
    gpt.decoder = torch.nn.Module()
    gpt.decoder.layers = torch.nn.ModuleList(
        [
            Layer(layer_number=4),
            Layer(layer_number=21),
        ]
    )
    float16_wrapper = torch.nn.Module()
    float16_wrapper.module = gpt
    ddp_wrapper = torch.nn.Module()
    ddp_wrapper.module = float16_wrapper

    wrapped = deepgemm_moe_forward.install_deepgemm_moe_forward(
        [ddp_wrapper],
        global_layer_indices=[20],
    )

    assert wrapped == ["module.module.decoder.layers.1.mlp.experts"]
    assert not hasattr(
        gpt.decoder.layers[0].mlp.experts,
        "_slime_deepgemm_moe_forward_wrapped",
    )
    assert gpt.decoder.layers[1].mlp.experts._slime_deepgemm_moe_forward_wrapped
    assert (
        deepgemm_moe_forward.install_deepgemm_moe_forward(
            [ddp_wrapper],
            global_layer_indices=[20],
        )
        == []
    )


def test_installer_skips_preallocated_workspace_when_unconfigured(monkeypatch, parallelism_one):
    del parallelism_one
    monkeypatch.delenv("SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES", raising=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 4
            self.mlp = torch.nn.Module()
            self.mlp.experts = TEGroupedMLP(num_experts=2)
            self.mlp.token_dispatcher = SimpleNamespace(num_local_experts=2)

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([Layer()])

    wrapped = deepgemm_moe_forward.install_deepgemm_moe_forward(
        model,
        global_layer_indices=[3],
    )

    dispatcher = model.decoder.layers[0].mlp.token_dispatcher
    experts = model.decoder.layers[0].mlp.experts
    assert wrapped == ["decoder.layers.0.mlp.experts"]
    assert not hasattr(dispatcher, deepgemm_moe_forward._COMBINE_WORKSPACE_ATTR)
    assert not hasattr(experts, deepgemm_moe_forward._COMBINE_WORKSPACE_ATTR)
    assert not hasattr(dispatcher, "_slime_preallocated_combine_wrapped")
    assert not hasattr(dispatcher, "_slime_preallocated_dispatch_wrapped")
    assert not hasattr(dispatcher, "_slime_preallocated_token_combine_wrapped")


def test_hook_uses_default_suffixes_when_cli_modules_are_none(monkeypatch):
    captured = {}

    def fake_install(model, layers, *, target_suffixes):
        captured.update(
            model=model,
            layers=layers,
            target_suffixes=target_suffixes,
        )

    monkeypatch.setattr(deepgemm_moe_forward, "install_deepgemm_moe_forward", fake_install)
    model = object()
    args = SimpleNamespace(
        megatron_deepgemm_moe_forward_layers=[3, 77],
        megatron_deepgemm_moe_forward_modules=None,
    )

    deepgemm_moe_forward.enable_deepgemm_moe_forward(args, model, store_prefix="actor_")

    assert captured == {
        "model": model,
        "layers": [3, 77],
        "target_suffixes": ("mlp.experts",),
    }


def _require_cuda_deepgemm(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real DeepGEMM diff tests")
    monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    for module_name in (
        "deep_gemm",
        "sglang.srt.layers.deep_gemm_wrapper",
        "sgl_kernel",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            pytest.skip(f"{module_name} is unavailable: {exc}")
    try:
        sglang_compat.import_sglang_module(
            "sglang.kernels.ops.quantization.fp8_kernel",
            "sglang.srt.layers.quantization.fp8_kernel",
        )
    except Exception as exc:
        pytest.skip(f"SGLang FP8 kernel is unavailable: {exc}")


def test_cuda_deepgemm_moe_forward_diff_against_bf16_reference_with_expert_m_padding(
    monkeypatch,
):
    _require_cuda_deepgemm(monkeypatch)
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_expert_tensor_parallel_world_size",
        lambda: 1,
    )
    torch.cuda.set_device(0)
    torch.manual_seed(456)
    module = TEGroupedMLP(num_experts=3, hidden_size=128, ffn_hidden_size=128).cuda().to(torch.bfloat16)
    tokens_per_expert = torch.tensor([129, 17, 260], device="cuda", dtype=torch.int32)
    num_tokens = int(tokens_per_expert.sum().item())
    hidden_states = (torch.randn(num_tokens, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)
    permuted_probs = torch.rand(num_tokens, device="cuda", dtype=torch.float32) * 1.5 + 0.1
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )

    output = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden_states,
        tokens_per_expert,
        permuted_probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
    )
    reference = _post_fc2_probability_reference(
        module,
        hidden_states,
        tokens_per_expert,
        permuted_probs,
    )
    diff = (output.float() - reference.float()).abs()

    assert diff.mean().item() < 0.02
    assert diff.max().item() < 0.20


def test_cuda_deepgemm_moe_expert_grouping_is_bitwise_batch_invariant(
    monkeypatch,
):
    _require_cuda_deepgemm(monkeypatch)
    monkeypatch.setenv("SGLANG_DEEPGEMM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_expert_tensor_parallel_world_size",
        lambda: 1,
    )
    torch.cuda.set_device(0)
    torch.manual_seed(458)
    module = TEGroupedMLP(num_experts=3, hidden_size=128, ffn_hidden_size=128).cuda().to(torch.bfloat16)
    tokens_per_expert = torch.tensor([129, 17, 260], device="cuda", dtype=torch.int32)
    num_tokens = int(tokens_per_expert.sum().item())
    hidden_states = (torch.randn(num_tokens, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)
    permuted_probs = torch.rand(num_tokens, device="cuda", dtype=torch.float32) * 1.5 + 0.1
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )

    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "3")
    ungrouped = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden_states,
        tokens_per_expert,
        permuted_probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
    )
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "1")
    grouped = deepgemm_moe_forward._deepgemm_grouped_moe_forward(
        module,
        hidden_states,
        tokens_per_expert,
        permuted_probs,
        layout=layout,
        module_name="decoder.layers.3.mlp.experts",
    )

    torch.testing.assert_close(grouped, ungrouped, rtol=0, atol=0)


def test_cuda_deepgemm_moe_backward_diff_against_bf16_reference_with_expert_m_padding(
    monkeypatch,
):
    _require_cuda_deepgemm(monkeypatch)
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_expert_tensor_parallel_world_size",
        lambda: 1,
    )
    torch.cuda.set_device(0)
    torch.manual_seed(457)

    module = TEGroupedMLP(num_experts=3, hidden_size=128, ffn_hidden_size=128).cuda().to(torch.bfloat16)
    reference_module = TEGroupedMLP(num_experts=3, hidden_size=128, ffn_hidden_size=128).cuda().to(torch.bfloat16)
    reference_module.load_state_dict(module.state_dict())
    tokens_per_expert = torch.tensor(
        [129, 17, 260],
        device="cuda",
        dtype=torch.int32,
    )
    num_tokens = int(tokens_per_expert.sum().item())
    hidden_states = (
        (
            torch.randn(
                num_tokens,
                128,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.2
        )
        .to(torch.bfloat16)
        .requires_grad_()
    )
    permuted_probs = (torch.rand(num_tokens, device="cuda", dtype=torch.float32) * 1.5 + 0.1).requires_grad_()
    grad_output = (
        torch.randn(
            num_tokens,
            128,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.2
    ).to(torch.bfloat16)

    reference_hidden = hidden_states.detach().clone().requires_grad_()
    reference_probs = permuted_probs.detach().clone().requires_grad_()
    reference_output = _post_fc2_probability_reference(
        reference_module,
        reference_hidden,
        tokens_per_expert,
        reference_probs,
    )
    reference_output.backward(grad_output)

    assert deepgemm_moe_forward._wrap_te_grouped_mlp(
        module,
        "decoder.layers.3.mlp.experts",
    )
    output, _ = module(hidden_states, tokens_per_expert, permuted_probs)
    assert module.forward_calls == 0
    output.backward(grad_output)

    checks = {
        "output": (output, reference_output, 0.02, 0.20),
        "hidden_grad": (
            hidden_states.grad,
            reference_hidden.grad,
            0.02,
            0.20,
        ),
        "router_probability_grad": (
            permuted_probs.grad,
            reference_probs.grad,
            0.03,
            0.30,
        ),
    }
    for (name, parameter), (_, reference_parameter) in zip(
        module.named_parameters(),
        reference_module.named_parameters(),
        strict=True,
    ):
        checks[f"{name}_grad"] = (
            parameter.grad,
            reference_parameter.grad,
            0.02,
            0.20,
        )

    for name, (actual, reference, mean_limit, max_limit) in checks.items():
        diff = (actual.float() - reference.float()).abs()
        assert diff.mean().item() < mean_limit, f"{name} mean diff {diff.mean().item()} >= {mean_limit}"
        assert diff.max().item() < max_limit, f"{name} max diff {diff.max().item()} >= {max_limit}"


def test_cuda_grouped_bf16_trainable_expert_backward_matches_per_expert_path(
    monkeypatch,
):
    _require_cuda_deepgemm(monkeypatch)
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        deepgemm_moe_forward.parallel_state,
        "get_expert_tensor_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_BF16_BACKWARD_EXPERTS_PER_GROUP", "3")
    # Force adaptive grouping into [expert 0, expert 1] and [expert 2] while
    # keeping every individual expert below the cap.
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_BF16_BACKWARD_MAX_PADDED_BYTES", "131072")
    torch.cuda.set_device(0)
    torch.manual_seed(459)

    old_module = TEGroupedMLP(num_experts=3, hidden_size=128, ffn_hidden_size=128).cuda().to(torch.bfloat16)
    layout = deepgemm_moe_forward._validate_te_grouped_mlp(
        old_module,
        "decoder.layers.3.mlp.experts",
    )
    assert deepgemm_moe_forward._wrap_te_grouped_mlp(
        old_module,
        "decoder.layers.3.mlp.experts",
    )

    tokens_per_expert = torch.tensor([129, 17, 260], device="cuda", dtype=torch.int32)
    num_tokens = int(tokens_per_expert.sum().item())
    hidden = (torch.randn(num_tokens, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)
    probs = torch.rand(num_tokens, device="cuda", dtype=torch.float32) * 1.5 + 0.1
    grad_output = (torch.randn(num_tokens, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)

    monkeypatch.delenv("SLIME_DEEPGEMM_MOE_GROUPED_BF16_BACKWARD", raising=False)
    assert deepgemm_moe_forward._use_grouped_bf16_backward(
        hidden,
        (129, 17, 260),
        (True,) * layout.num_local_experts,
        (True,) * layout.num_local_experts,
    )

    old_hidden = hidden.detach().clone().requires_grad_()
    old_probs = probs.detach().clone().requires_grad_()
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_GROUPED_BF16_BACKWARD", "0")
    old_output, _ = old_module(old_hidden, tokens_per_expert, old_probs)
    old_output.backward(grad_output)
    old_hidden_grad = old_hidden.grad.detach().clone()
    old_probs_grad = old_probs.grad.detach().clone()
    old_fc1_grads = tuple(
        getattr(old_module.linear_fc1, f"weight{expert_index}").grad.detach().clone()
        for expert_index in range(layout.num_local_experts)
    )
    old_fc2_grads = tuple(
        getattr(old_module.linear_fc2, f"weight{expert_index}").grad.detach().clone()
        for expert_index in range(layout.num_local_experts)
    )

    grouped_hidden_grad = torch.empty_like(hidden)
    grouped_probs_grad = torch.empty_like(probs)
    fc1_weights = tuple(
        getattr(old_module.linear_fc1, f"weight{expert_index}").detach()
        for expert_index in range(layout.num_local_experts)
    )
    fc2_weights = tuple(
        getattr(old_module.linear_fc2, f"weight{expert_index}").detach()
        for expert_index in range(layout.num_local_experts)
    )
    (
        grouped_hidden_grad,
        grouped_probs_grad,
        grouped_fc1_grads,
        grouped_fc2_grads,
    ) = deepgemm_moe_forward._grouped_expert_backward(
        hidden_states=hidden,
        permuted_probs=probs,
        grad_output=grad_output,
        fc1_weights=fc1_weights,
        fc2_weights=fc2_weights,
        counts=(129, 17, 260),
        layout=layout,
        needs_hidden=True,
        needs_probs=True,
        needs_fc1_weights=(True,) * layout.num_local_experts,
        needs_fc2_weights=(True,) * layout.num_local_experts,
        defer_router_probabilities=False,
        grad_hidden=grouped_hidden_grad,
        grad_probs=grouped_probs_grad,
    )

    torch.testing.assert_close(grouped_hidden_grad, old_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(grouped_probs_grad, old_probs.grad, rtol=0, atol=0)
    for expert_index in range(layout.num_local_experts):
        fc1_weight = getattr(old_module.linear_fc1, f"weight{expert_index}")
        fc2_weight = getattr(old_module.linear_fc2, f"weight{expert_index}")
        torch.testing.assert_close(grouped_fc1_grads[expert_index], fc1_weight.grad, rtol=0, atol=0)
        torch.testing.assert_close(grouped_fc2_grads[expert_index], fc2_weight.grad, rtol=0, atol=0)

    # Once a grouped expert range has produced both wgrads, its input values
    # are dead and can hold dgrad without changing any gradient bits.
    reused_hidden_grad = hidden.clone()
    reused_probs_grad = torch.empty_like(probs)
    reused_ptr = reused_hidden_grad.data_ptr()
    (
        reused_hidden_grad,
        reused_probs_grad,
        reused_fc1_grads,
        reused_fc2_grads,
    ) = deepgemm_moe_forward._grouped_expert_backward(
        hidden_states=reused_hidden_grad,
        permuted_probs=probs,
        grad_output=grad_output,
        fc1_weights=fc1_weights,
        fc2_weights=fc2_weights,
        counts=(129, 17, 260),
        layout=layout,
        needs_hidden=True,
        needs_probs=True,
        needs_fc1_weights=(True,) * layout.num_local_experts,
        needs_fc2_weights=(True,) * layout.num_local_experts,
        defer_router_probabilities=False,
        grad_hidden=reused_hidden_grad,
        grad_probs=reused_probs_grad,
    )

    assert reused_hidden_grad.data_ptr() == reused_ptr
    torch.testing.assert_close(reused_hidden_grad, grouped_hidden_grad, rtol=0, atol=0)
    torch.testing.assert_close(reused_probs_grad, grouped_probs_grad, rtol=0, atol=0)
    for expert_index in range(layout.num_local_experts):
        torch.testing.assert_close(reused_fc1_grads[expert_index], grouped_fc1_grads[expert_index], rtol=0, atol=0)
        torch.testing.assert_close(reused_fc2_grads[expert_index], grouped_fc2_grads[expert_index], rtol=0, atol=0)

    # A hot expert makes the production path fall back to per-expert chunks.
    # That path must also finish fc1 wgrad's read before overwriting each input
    # chunk with dgrad, otherwise formal all-parameter training is corrupted.
    old_module.zero_grad(set_to_none=True)
    fallback_hidden = hidden.detach().clone().requires_grad_()
    fallback_probs = probs.detach().clone().requires_grad_()
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_GROUPED_BF16_BACKWARD", "0")
    old_module._slime_reuse_expert_input_for_grad = True
    fallback_output, _ = old_module(fallback_hidden, tokens_per_expert, fallback_probs)
    fallback_output.backward(grad_output)

    # The saved input storage itself is the dgrad destination; this assertion
    # would fail if the fallback silently allocated a separate full tensor.
    torch.testing.assert_close(fallback_hidden.detach(), old_hidden_grad, rtol=0, atol=0)
    torch.testing.assert_close(fallback_hidden.grad, old_hidden_grad, rtol=0, atol=0)
    torch.testing.assert_close(fallback_probs.grad, old_probs_grad, rtol=0, atol=0)
    for expert_index in range(layout.num_local_experts):
        fc1_weight = getattr(old_module.linear_fc1, f"weight{expert_index}")
        fc2_weight = getattr(old_module.linear_fc2, f"weight{expert_index}")
        torch.testing.assert_close(fc1_weight.grad, old_fc1_grads[expert_index], rtol=0, atol=0)
        torch.testing.assert_close(fc2_weight.grad, old_fc2_grads[expert_index], rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
