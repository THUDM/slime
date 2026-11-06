"""CPU tests for R3 routing-replay accounting.

Covers the opt-in route-mismatch metric (the counting hook in
``get_routing_replay_compute_topk``, the real-row mask from
``get_routing_replay_real_rows``, the cross-rank reduction and the CI check
in ``model.train``) and ``RoutingReplay.assert_all_consumed``.
"""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401 - stub megatron.core.mpu before slime's megatron utils import it
import pytest
import torch
import torch.distributed as dist

# data.py builds PackedSeqParams; the CPU image has no megatron.
_packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
_packed_seq_params.PackedSeqParams = SimpleNamespace
sys.modules.setdefault("megatron.core.packed_seq_params", _packed_seq_params)

from megatron.core import mpu  # noqa: E402

from slime.backends.megatron_utils.cp_utils import prepare_routed_experts_for_routing_replay  # noqa: E402
from slime.backends.megatron_utils.data import DataIterator, get_batch, get_routing_replay_real_rows  # noqa: E402
from slime.utils import accelerator, routing_replay  # noqa: E402
from slime.utils.routing_replay import (  # noqa: E402
    RoutingReplay,
    get_routing_replay_compute_topk,
    route_mismatch_metrics,
    router_recomputed,
    set_route_mismatch_rows,
    set_routing_replay,
)

NUM_GPUS = 0

NUM_EXPERTS = 8
TOPK = 2
NUM_LAYERS = 4
ALL_LAYERS = tuple(range(NUM_LAYERS))
PAD_MULTIPLIER = 2


def _set_layout(monkeypatch, *, cp_size=1, cp_rank=0, tp_size=1, tp_rank=0):
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: tp_size, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: tp_rank, raising=False)


@pytest.fixture(autouse=True)
def cpu_replay_runtime(monkeypatch):
    monkeypatch.setattr(accelerator, "current_device", lambda: "cpu")
    monkeypatch.setattr(accelerator, "device", lambda index=None: torch.device("cpu"))
    # Each simulated rank reduces alone; the tests sum the ranks themselves.
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, *args, **kwargs: None)
    # CPU-only torch cannot pin host memory, which the route layout helper requests.
    empty = torch.empty

    def unpinned(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", unpinned)
    monkeypatch.setattr(RoutingReplay, "all_routing_replays", [])
    monkeypatch.setattr(RoutingReplay, "lazy_resources", [])
    monkeypatch.setattr(routing_replay, "ROUTING_REPLAY", None)
    monkeypatch.setattr(routing_replay, "ROUTE_MISMATCH_ROWS", None)
    monkeypatch.setattr(routing_replay, "ORDERED_TOPK_CAPTURE_ROUTER", None)
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    _set_layout(monkeypatch)


def _topk(scores, topk, num_groups=None, group_topk=None):
    return torch.topk(scores, k=topk, dim=1)


def _samples(lengths, seed=0):
    """Per-sample token ids and per-token router scores [len, NUM_LAYERS, NUM_EXPERTS]."""
    generator = torch.Generator().manual_seed(seed)
    tokens = [torch.arange(1, n + 1) for n in lengths]
    scores = [torch.rand(n, NUM_LAYERS, NUM_EXPERTS, generator=generator) for n in lengths]
    return tokens, scores


def _natural(sample_scores):
    return sample_scores.topk(TOPK, dim=-1).indices


def _rollout_routes(scores):
    """Rollout routes agree with the trainer: its top-k on every token but the last."""
    return [_natural(sample_scores)[:-1].to(torch.int32) for sample_scores in scores]


def _brute_force(scores, routes):
    """Per layer: [real tokens, tokens whose replayed expert set differs from the natural top-k]."""
    real = sum(sample_routes.shape[0] for sample_routes in routes)
    mismatched = torch.zeros(NUM_LAYERS, dtype=torch.int64)
    for sample_scores, sample_routes in zip(scores, routes, strict=True):
        natural = _natural(sample_scores)[: sample_routes.shape[0]].sort(dim=-1).values
        mismatched += (natural != sample_routes.long().sort(dim=-1).values).any(dim=-1).sum(dim=0)
    return torch.stack([torch.full((NUM_LAYERS,), real), mismatched], dim=1)


def _run_rank(
    monkeypatch,
    tokens,
    scores,
    routes,
    microbatches,
    layers=ALL_LAYERS,
    *,
    cp_size=1,
    cp_rank=0,
    tp_size=1,
    tp_rank=0,
    allgather_cp=False,
    count_all_rows=False,
    seed=0,
):
    """Replay one rank's training forward through the real hook; return its reduced counts.

    The replay rows are laid out by ``prepare_routed_experts_for_routing_replay``. Router
    scores are placed with the same helper, from token ids no filler route can take, so the
    row -> token map is independent of the mask under test.
    """
    _set_layout(monkeypatch, cp_size=cp_size, cp_rank=cp_rank, tp_size=tp_size, tp_rank=tp_rank)
    sequence_parallel = tp_size > 1
    layout = dict(
        num_experts=NUM_EXPERTS,
        data_pad_size_multiplier=PAD_MULTIPLIER,
        sequence_parallel=sequence_parallel,
        allgather_cp=allgather_cp,
    )
    monkeypatch.setattr(RoutingReplay, "all_routing_replays", [])
    replays = {}
    for layer in layers:
        replays[layer] = RoutingReplay()
        replays[layer].layer_id = layer

    offsets = torch.tensor([0] + [len(t) for t in tokens]).cumsum(0)
    token_scores = torch.cat(scores)
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for microbatch in microbatches:
        mb_tokens = [tokens[i] for i in microbatch]
        replay_rows = prepare_routed_experts_for_routing_replay([routes[i] for i in microbatch], mb_tokens, **layout)
        token_ids = [(torch.arange(len(tokens[i]) - 1) + offsets[i] + NUM_EXPERTS).view(-1, 1, 1) for i in microbatch]
        row_ids = prepare_routed_experts_for_routing_replay(token_ids, mb_tokens, **layout)[:, 0, 0]
        from_rollout = row_ids >= NUM_EXPERTS
        row_scores = torch.rand(row_ids.shape[0], NUM_LAYERS, NUM_EXPERTS, generator=generator)
        row_scores[from_rollout] = token_scores[row_ids[from_rollout] - NUM_EXPERTS]
        for layer in layers:
            replays[layer].top_indices_list.append(replay_rows[:, layer])
        batches.append((mb_tokens, row_scores))

    compute_topk = get_routing_replay_compute_topk(_topk)
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
    for mb_tokens, row_scores in batches:
        real_rows = get_routing_replay_real_rows(mb_tokens, PAD_MULTIPLIER, allgather_cp, sequence_parallel)
        set_route_mismatch_rows(torch.ones_like(real_rows) if count_all_rows else real_rows)
        for layer in layers:
            set_routing_replay(replays[layer])
            compute_topk(row_scores[:, layer], TOPK)
        set_route_mismatch_rows(None)
    return RoutingReplay.reduce_route_mismatch(NUM_LAYERS)


def _sum_ranks(
    monkeypatch, tokens, scores, routes, *, dp_size=1, cp_size=1, allgather_cp=False, tp_size=1, pp_size=1, **kwargs
):
    """Sum the per-rank counts over a simulated DP x PP x CP x TP layout, as the all-reduce does."""
    total = torch.zeros(NUM_LAYERS, 3, dtype=torch.int64)
    for dp_rank in range(dp_size):
        samples = list(range(dp_rank, len(tokens), dp_size))
        microbatches = [samples[: len(samples) // 2], samples[len(samples) // 2 :]]
        for pp_rank in range(pp_size):
            layers = range(pp_rank * NUM_LAYERS // pp_size, (pp_rank + 1) * NUM_LAYERS // pp_size)
            for cp_rank in range(cp_size):
                for tp_rank in range(tp_size):
                    total += _run_rank(
                        monkeypatch,
                        tokens,
                        scores,
                        routes,
                        microbatches,
                        layers,
                        cp_size=cp_size,
                        cp_rank=cp_rank,
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                        allgather_cp=allgather_cp,
                        seed=dp_rank * 1000 + pp_rank * 100 + cp_rank * 10 + tp_rank,
                        **kwargs,
                    )
    return total


LENGTHS = [5, 9, 2, 14, 7, 1, 11, 3]


# ---------------------------------------------------------------------------
# Real-row mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_real_rows_are_the_rows_that_replay_rollout_routes(monkeypatch, cp_size, allgather_cp, sequence_parallel):
    lengths = [1, 2, 7, 33, 128]
    # Token ids encode (sample, position); 0 is get_batch's pad token.
    tokens = [torch.arange(1, n + 1) + 1000 * sample for sample, n in enumerate(lengths)]
    last_tokens = torch.tensor([1000 * sample + n for sample, n in enumerate(lengths)])
    # A route value no filler route can take marks the rows copied from the rollout.
    routes = [torch.full((n - 1, 1, 1), NUM_EXPERTS, dtype=torch.int32) for n in lengths]
    rollout_data = {
        "tokens": tokens,
        "loss_masks": [torch.ones(n - 1, dtype=torch.int) for n in lengths],
        "total_lengths": lengths,
        "response_lengths": [n - 1 for n in lengths],
    }
    tp_size = 2 if sequence_parallel else 1
    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            _set_layout(monkeypatch, cp_size=cp_size, cp_rank=cp_rank, tp_size=tp_size, tp_rank=tp_rank)
            real_rows = get_routing_replay_real_rows(tokens, PAD_MULTIPLIER, allgather_cp, sequence_parallel)
            assert real_rows.dtype == torch.bool

            replay_rows = prepare_routed_experts_for_routing_replay(
                routes,
                tokens,
                num_experts=NUM_EXPERTS,
                data_pad_size_multiplier=PAD_MULTIPLIER,
                sequence_parallel=sequence_parallel,
                allgather_cp=allgather_cp,
            )
            assert torch.equal(real_rows, replay_rows[:, 0, 0] == NUM_EXPERTS)

            batch = get_batch(
                DataIterator(rollout_data, [list(range(len(lengths)))]),
                ["tokens", "loss_masks", "total_lengths", "response_lengths"],
                PAD_MULTIPLIER,
                allgather_cp,
            )
            # Megatron's sequence-parallel scatter splits the token stream into equal TP chunks.
            token_rows = batch["tokens"][0].chunk(tp_size)[tp_rank]
            assert torch.equal(real_rows, (token_rows != 0) & ~torch.isin(token_rows, last_tokens))


# ---------------------------------------------------------------------------
# Route-mismatch metric
# ---------------------------------------------------------------------------


def test_identical_routes_count_every_real_token_and_no_mismatch(monkeypatch):
    tokens, scores = _samples(LENGTHS)
    routes = _rollout_routes(scores)

    counts = _sum_ranks(monkeypatch, tokens, scores, routes)

    real_tokens = sum(n - 1 for n in LENGTHS)
    assert counts.tolist() == [[1, real_tokens, 0]] * NUM_LAYERS


def test_one_token_shift_matches_brute_force_per_layer(monkeypatch):
    tokens, scores = _samples(LENGTHS)
    # Each real token replays the previous token's route (a known row permutation).
    routes = [sample_routes.roll(1, dims=0) for sample_routes in _rollout_routes(scores)]

    counts = _sum_ranks(monkeypatch, tokens, scores, routes)

    expected = _brute_force(scores, routes)
    assert torch.equal(counts[:, 1:], expected)
    assert (expected[:, 1] > 0).all()


def test_synthetic_rows_are_excluded(monkeypatch):
    tokens, scores = _samples(LENGTHS)
    routes = _rollout_routes(scores)
    layout = dict(dp_size=1, cp_size=2, tp_size=2)

    counts = _sum_ranks(monkeypatch, tokens, scores, routes, **layout)
    every_row = _sum_ranks(monkeypatch, tokens, scores, routes, count_all_rows=True, **layout)

    real_tokens = sum(n - 1 for n in LENGTHS)
    assert counts[:, 1:].tolist() == [[real_tokens, 0]] * NUM_LAYERS
    # Control: sample tails and CP/TP padding replay filler routes, which disagree with the router.
    assert (every_row[:, 1] > real_tokens).all()
    assert (every_row[:, 2] > 0).all()


@pytest.mark.parametrize("ordered_topk_capture", [False, True])
def test_same_expert_set_in_another_order_is_agreement(monkeypatch, ordered_topk_capture):
    tokens, scores = _samples(LENGTHS)
    routes = [sample_routes.flip(-1) for sample_routes in _rollout_routes(scores)]
    capture_router = SimpleNamespace()
    if ordered_topk_capture:
        # GLM-5's DeepEP alignment path takes an unsorted top-k.
        monkeypatch.setattr(routing_replay, "ORDERED_TOPK_CAPTURE_ROUTER", capture_router)

    counts = _sum_ranks(monkeypatch, tokens, scores, routes)

    assert counts[:, 2].tolist() == [0] * NUM_LAYERS
    if ordered_topk_capture:
        # The capture keeps the replayed order, never the comparison's top-k.
        last_replayed = routes[LENGTHS.index(3)]  # the last microbatch's last sample
        assert torch.equal(capture_router._slime_ordered_topk_indices[-3:-1], last_replayed[:, NUM_LAYERS - 1])


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("cp_size,allgather_cp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_counts_match_brute_force_for_every_layout(monkeypatch, dp_size, cp_size, allgather_cp, tp_size, pp_size):
    tokens, scores = _samples(LENGTHS, seed=1)
    routes = _rollout_routes(scores)
    for sample, sample_routes in enumerate(routes):
        for position in range(sample_routes.shape[0]):
            for layer in range(NUM_LAYERS):
                if (sample + 2 * position + layer) % 5 == 0:
                    # A different expert set: swap one routed expert for an unrouted one.
                    routed = sample_routes[position, layer].tolist()
                    sample_routes[position, layer, 0] = min(set(range(NUM_EXPERTS)) - set(routed))
                elif (sample + position + layer) % 3 == 0:
                    sample_routes[position, layer] = sample_routes[position, layer].flip(0)

    counts = _sum_ranks(
        monkeypatch,
        tokens,
        scores,
        routes,
        dp_size=dp_size,
        cp_size=cp_size,
        allgather_cp=allgather_cp,
        tp_size=tp_size,
        pp_size=pp_size,
    )

    # Each layer's router exists on every DP x CP x TP rank of the stage that owns it.
    assert counts[:, 0].tolist() == [dp_size * cp_size * tp_size] * NUM_LAYERS
    expected = _brute_force(scores, routes)
    assert torch.equal(counts[:, 1:], expected)
    assert (expected[:, 1] > 0).all()


def _replays_for_one_rank(tokens, scores, routes, microbatches):
    replays = []
    rows = []
    for layer in range(NUM_LAYERS):
        replay = RoutingReplay()
        replay.layer_id = layer
        replays.append(replay)
    layout = dict(
        num_experts=NUM_EXPERTS, data_pad_size_multiplier=PAD_MULTIPLIER, sequence_parallel=False, allgather_cp=False
    )
    for microbatch in microbatches:
        mb_tokens = [tokens[i] for i in microbatch]
        replay_rows = prepare_routed_experts_for_routing_replay([routes[i] for i in microbatch], mb_tokens, **layout)
        for layer, replay in enumerate(replays):
            replay.top_indices_list.append(replay_rows[:, layer])
        mb_scores = torch.cat([scores[i] for i in microbatch])
        padded = torch.zeros(replay_rows.shape[0], NUM_LAYERS, NUM_EXPERTS)
        padded[: mb_scores.shape[0]] = mb_scores
        rows.append((mb_tokens, padded))
    return replays, rows


def test_only_the_training_forward_counts(monkeypatch):
    tokens, scores = _samples(LENGTHS)
    routes = [sample_routes.roll(1, dims=0) for sample_routes in _rollout_routes(scores)]
    microbatches = [[0, 1, 2, 3], [4, 5, 6, 7]]
    replays, batches = _replays_for_one_rank(tokens, scores, routes, microbatches)
    compute_topk = get_routing_replay_compute_topk(_topk)

    def run_routers(row_scores):
        for layer, replay in enumerate(replays):
            set_routing_replay(replay)
            compute_topk(row_scores[:, layer], TOPK)

    # Actor log-prob pass: replay_forward is set directly and no real rows are published.
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
    for _, row_scores in batches:
        run_routers(row_scores)
    assert all(replay.route_mismatch is None for replay in replays)
    RoutingReplay.clear_all_forward()

    for mb_tokens, row_scores in batches:
        # Training forward (forward_step flips replay_backward -> replay_forward).
        monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
        set_route_mismatch_rows(get_routing_replay_real_rows(mb_tokens, PAD_MULTIPLIER, False, False))
        run_routers(row_scores)
        # Activation recompute re-runs the routers in backward; even with rows published it never counts.
        monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_backward")
        run_routers(row_scores)
        set_route_mismatch_rows(None)

    counts = RoutingReplay.reduce_route_mismatch(NUM_LAYERS)
    assert torch.equal(counts[:, 1:], _brute_force(scores, routes))
    for replay in replays:
        replay.recomputed = True
    RoutingReplay.assert_all_consumed()


@pytest.mark.parametrize("ordered_topk_capture", [False, True])
def test_metric_off_runs_no_extra_topk_and_leaves_the_replay_unchanged(monkeypatch, ordered_topk_capture):
    tokens, scores = _samples(LENGTHS)
    routes = [sample_routes.roll(1, dims=0) for sample_routes in _rollout_routes(scores)]
    replays, batches = _replays_for_one_rank(tokens, scores, routes, [list(range(len(LENGTHS)))])
    mb_tokens, row_scores = batches[0]
    row_scores.requires_grad_(True)
    router_topk_scores = []
    router_topk = routing_replay._compute_topk_for_current_router

    def counting_router_topk(old_compute_topk, scores, *args, **kwargs):
        router_topk_scores.append(scores)
        return router_topk(old_compute_topk, scores, *args, **kwargs)

    # Every top-k the router itself runs, on the Megatron and the GLM-5 unsorted paths alike.
    monkeypatch.setattr(routing_replay, "_compute_topk_for_current_router", counting_router_topk)
    compute_topk = get_routing_replay_compute_topk(_topk)
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
    capture_router = SimpleNamespace() if ordered_topk_capture else None
    monkeypatch.setattr(routing_replay, "ORDERED_TOPK_CAPTURE_ROUTER", capture_router)

    def run(real_rows):
        RoutingReplay.clear_all_forward()
        set_route_mismatch_rows(real_rows)
        outputs = []
        for layer, replay in enumerate(replays):
            set_routing_replay(replay)
            probs, top_indices = compute_topk(row_scores[:, layer], TOPK)
            if capture_router is not None:
                assert capture_router._slime_ordered_topk_indices is top_indices
            outputs.append((probs, top_indices))
        set_route_mismatch_rows(None)
        return outputs

    off = run(None)
    assert router_topk_scores == []
    assert all(replay.route_mismatch is None for replay in replays)

    on = run(get_routing_replay_real_rows(mb_tokens, PAD_MULTIPLIER, False, False))
    assert len(router_topk_scores) == NUM_LAYERS
    assert all(not scores_arg.requires_grad for scores_arg in router_topk_scores)
    for (probs_off, indices_off), (probs_on, indices_on) in zip(off, on, strict=True):
        assert torch.equal(indices_off, indices_on)
        assert torch.equal(probs_off, probs_on)
        assert probs_on.grad_fn is not None
    for replay in replays:
        assert not replay.route_mismatch.requires_grad
        assert replay.route_mismatch.grad_fn is None


def test_route_mismatch_metrics_summarize_moe_layers():
    # Columns: routers, real tokens, mismatched tokens. Layers 0 and 4 are dense.
    counts = torch.tensor([[0, 0, 0], [2, 10, 1], [2, 10, 3], [2, 10, 0], [0, 0, 0]])

    metrics = route_mismatch_metrics(counts)

    assert metrics == pytest.approx(
        {
            "r3_route_mismatch_min_tokens": 10,
            "r3_route_mismatch": (0.1 + 0.3 + 0.0) / 3,
            "r3_route_mismatch_max": 0.3,
            "r3_route_mismatch_layer_1": 0.1,
            "r3_route_mismatch_layer_2": 0.3,
            "r3_route_mismatch_layer_3": 0.0,
        }
    )


def _r3_ci_check():
    """The R3 CI assertions from ``model.train``, compiled on their own."""
    source = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/model.py"
    train = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    (check,) = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.ci_test and route_mismatch is not None"
    ]
    return compile(ast.fix_missing_locations(ast.Module(body=[check], type_ignores=[])), "model.py", "exec")


@pytest.mark.parametrize(
    "threshold,log_dict,passes",
    [
        (0.2, {"train/r3_route_mismatch_min_tokens": 100, "train/r3_route_mismatch_max": 0.2}, True),
        (None, {"train/r3_route_mismatch_min_tokens": 100, "train/r3_route_mismatch_max": 0.9}, True),
        (0.2, {"train/r3_route_mismatch_min_tokens": 100, "train/r3_route_mismatch_max": 0.21}, False),
        # Zero mismatch is legitimate (deterministic rollouts), zero replayed tokens is not.
        (0.2, {"train/r3_route_mismatch_min_tokens": 100, "train/r3_route_mismatch_max": 0.0}, True),
        (0.2, {"train/r3_route_mismatch_min_tokens": 0, "train/r3_route_mismatch_max": 0.0}, False),
        (None, {}, False),
    ],
)
def test_ci_check_requires_replayed_tokens_and_bounds_the_worst_layer(threshold, log_dict, passes):
    namespace = {
        "args": SimpleNamespace(ci_test=True, ci_r3_route_mismatch_threshold=threshold),
        "log_dict": log_dict,
        "route_mismatch": torch.zeros(NUM_LAYERS, 3, dtype=torch.int64),
        "role_tag": "",
    }
    if passes:
        exec(_r3_ci_check(), namespace)
    else:
        with pytest.raises(AssertionError):
            exec(_r3_ci_check(), namespace)


def test_route_mismatch_metrics_flag_a_layer_that_counted_nothing():
    counts = torch.tensor([[2, 10, 1], [2, 10, 3], [2, 0, 0]])

    metrics = route_mismatch_metrics(counts)

    assert metrics["r3_route_mismatch_min_tokens"] == 0
    assert "r3_route_mismatch_layer_2" not in metrics
    assert metrics["r3_route_mismatch_max"] == pytest.approx(0.3)
    assert route_mismatch_metrics(torch.zeros(3, 3, dtype=torch.int64)) == {}


# ---------------------------------------------------------------------------
# assert_all_consumed
# ---------------------------------------------------------------------------


def _config(granularity=None, method=None, num_layers=None, modules=None):
    return SimpleNamespace(
        recompute_granularity=granularity,
        recompute_method=method,
        recompute_num_layers=num_layers,
        recompute_modules=modules,
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        (_config(), [False, False, False, False]),
        (_config("full", "uniform", 1), [True, True, True, True]),
        (_config("full", "uniform", 4), [True, True, True, True]),
        (_config("full", "block", 2), [True, True, False, False]),
        (_config("selective", modules=["core_attn"]), [False, False, False, False]),
        (_config("selective", modules=["moe"]), [True, True, True, True]),
        (_config("selective", modules=["moe_act", "layernorm", "shared_experts"]), [False, False, False, False]),
        (_config("selective", modules=["mlp"]), [False, False, False, False]),
    ],
)
def test_router_recomputed_follows_megatron_recompute(config, expected):
    assert [router_recomputed(config, layer_index) for layer_index in range(4)] == expected


def _consume(recomputed, forward_pops, backward_pops, recorded=3):
    for layer, (layer_recomputed, num_forward, num_backward) in enumerate(
        zip(recomputed, forward_pops, backward_pops, strict=True)
    ):
        replay = RoutingReplay()
        replay.layer_id = layer
        replay.recomputed = layer_recomputed
        replay.top_indices_list = [torch.zeros(4, TOPK, dtype=torch.int32) for _ in range(recorded)]
        for _ in range(num_forward):
            replay.pop_forward()
        for _ in range(num_backward):
            replay.pop_backward()


@pytest.mark.parametrize(
    "recomputed,backward_pops",
    [
        ([True] * 4, [3] * 4),  # full recompute
        ([False] * 4, [0] * 4),  # no recompute
        ([True, True, False, False], [3, 3, 0, 0]),  # partial (block) recompute
    ],
)
def test_assert_all_consumed_passes_for_the_active_recompute(recomputed, backward_pops):
    _consume(recomputed, [3] * 4, backward_pops)

    RoutingReplay.assert_all_consumed()


@pytest.mark.parametrize(
    "recomputed,forward_pops,backward_pops,match",
    [
        ([True] * 4, [3, 2, 3, 3], [3] * 4, r"layer 1.*forward=2"),  # missing forward pop
        ([True] * 4, [3] * 4, [3, 3, 2, 3], r"layer 2.*backward=2"),  # missing backward pop
        ([True, True, False, False], [3] * 4, [3, 3, 3, 0], r"layer 2.*backward=3"),  # extra pop: not recomputed
        ([True, True, False, False], [3] * 4, [3, 0, 0, 0], r"layer 1.*backward=0"),  # recomputed, never replayed
    ],
)
def test_assert_all_consumed_fails_on_a_missing_or_extra_pop(recomputed, forward_pops, backward_pops, match):
    _consume(recomputed, forward_pops, backward_pops)

    with pytest.raises(RuntimeError, match=match):
        RoutingReplay.assert_all_consumed()


def test_one_token_shift_passes_consumption_but_exceeds_the_mismatch_bound(monkeypatch):
    tokens, scores = _samples(LENGTHS)
    routes = [sample_routes.roll(1, dims=0) for sample_routes in _rollout_routes(scores)]
    replays, batches = _replays_for_one_rank(tokens, scores, routes, [[0, 1, 2, 3], [4, 5, 6, 7]])
    compute_topk = get_routing_replay_compute_topk(_topk)
    for replay in replays:
        replay.recomputed = True
    for mb_tokens, row_scores in batches:
        for stage in ("replay_forward", "replay_backward"):
            monkeypatch.setenv("ROUTING_REPLAY_STAGE", stage)
            set_route_mismatch_rows(get_routing_replay_real_rows(mb_tokens, PAD_MULTIPLIER, False, False))
            for layer, replay in enumerate(replays):
                set_routing_replay(replay)
                compute_topk(row_scores[:, layer], TOPK)
            set_route_mismatch_rows(None)

    RoutingReplay.assert_all_consumed()
    metrics = route_mismatch_metrics(RoutingReplay.reduce_route_mismatch(NUM_LAYERS))
    assert metrics["r3_route_mismatch_max"] > 0.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
