import os
import torch
import torch.distributed as dist

from slime.utils import accelerator


ROUTING_REPLAY = None
ORDERED_TOPK_CAPTURE_ROUTER = None
# Real-row mask of the microbatch in the training forward; route mismatches are counted only while set.
ROUTE_MISMATCH_ROWS = None


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


def set_route_mismatch_rows(rows):
    global ROUTE_MISMATCH_ROWS
    ROUTE_MISMATCH_ROWS = rows


def _set_ordered_topk_capture_router(router):
    global ORDERED_TOPK_CAPTURE_ROUTER
    ORDERED_TOPK_CAPTURE_ROUTER = router


def _capture_ordered_topk(top_indices):
    """Keep the current router's exact top-k order until its MoE combine."""
    router = ORDERED_TOPK_CAPTURE_ROUTER
    if router is not None:
        router._slime_ordered_topk_indices = top_indices


def consume_ordered_topk(module):
    """Return and release the current forward's ordered top-k indices."""
    return module.__dict__.pop("_slime_ordered_topk_indices", None)


def register_ordered_topk_capture(module):
    """Capture one forward's SGLang-compatible top-k order without R3."""
    if getattr(module, "_slime_ordered_topk_capture_registered", False):
        return

    _install_router_replay_bridge(module)

    def pre_forward_hook(patched_module, *args, **kwargs):
        del args, kwargs
        _set_ordered_topk_capture_router(patched_module)

    def forward_hook(patched_module, *args, **kwargs):
        del args, kwargs
        if ORDERED_TOPK_CAPTURE_ROUTER is patched_module:
            _set_ordered_topk_capture_router(None)

    module.register_forward_pre_hook(pre_forward_hook)
    module.register_forward_hook(forward_hook)
    module._slime_ordered_topk_capture_registered = True


def _compute_topk_for_current_router(
    old_compute_topk,
    scores,
    topk,
    num_groups=None,
    group_topk=None,
):
    # SGLang's deterministic DeepSeek/GLM biased top-k uses
    # torch.topk(..., sorted=False).  Megatron's local compute_topk uses the
    # default sorted=True.  The selected expert set is the same, but the
    # low-latency-compatible owner reduction consumes experts in top-k column
    # order, so the difference changes BF16 accumulation before any route
    # actually diverges.
    #
    # Only override routers registered by the DeepEP alignment bridge, and only for the
    # non-grouped Megatron path used by GLM-5 (n_group=topk_group=1 in SGLang,
    # represented as no group limit in Megatron).  Other training paths retain
    # Megatron's original semantics.
    if ORDERED_TOPK_CAPTURE_ROUTER is not None and not group_topk:
        return torch.topk(scores, k=topk, dim=1, sorted=False)

    return old_compute_topk(
        scores,
        topk,
        num_groups=num_groups,
        group_topk=group_topk,
    )


def router_recomputed(config, layer_index: int) -> bool:
    """Whether activation recompute re-runs a MoE layer's router, popping its replay in backward.

    ``layer_index`` is the layer's position in its (virtual) pipeline chunk. Megatron's full
    recompute checkpoints every layer ("uniform") or the first ``recompute_num_layers``
    ("block"; under FP8/FP4 that window shifts past stage inputs that need no grad), and
    selective recompute re-runs the router only for the "moe" module.
    """
    if config.recompute_granularity == "full":
        return config.recompute_method == "uniform" or layer_index < config.recompute_num_layers
    return config.recompute_granularity == "selective" and "moe" in config.recompute_modules


def route_mismatch_metrics(counts: torch.Tensor) -> dict[str, float]:
    """Summarize ``RoutingReplay.reduce_route_mismatch`` over the MoE layers, i.e. rows with routers."""
    moe_layers = [
        (layer_id, real, mismatched)
        for layer_id, (routers, real, mismatched) in enumerate(counts.tolist())
        if routers > 0
    ]
    if not moe_layers:
        return {}
    rates = {layer_id: mismatched / real for layer_id, real, mismatched in moe_layers if real > 0}
    # Every MoE layer routes the same tokens, so a layer with fewer real tokens missed replays.
    metrics = {"r3_route_mismatch_min_tokens": min(real for _, real, _ in moe_layers)}
    if rates:
        metrics["r3_route_mismatch"] = sum(rates.values()) / len(rates)
        metrics["r3_route_mismatch_max"] = max(rates.values())
        metrics.update({f"r3_route_mismatch_layer_{layer_id}": rate for layer_id, rate in rates.items()})
    return metrics


class RoutingReplay:
    all_routing_replays = []
    lazy_resources = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        # Bound by fill_routing_replay: the global layer id, and whether recompute re-runs this router.
        self.layer_id = None
        self.recomputed = None
        # [real tokens, mismatched tokens] counted on device since the last reduction.
        self.route_mismatch = None
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        if hasattr(top_indices, "materialize_for_routing_replay"):
            self.top_indices_list.append(top_indices)
            return
        if top_indices.device.type == "cpu" and top_indices.is_pinned() and top_indices.is_contiguous():
            self.top_indices_list.append(top_indices)
            return

        # Compact non-contiguous layer views so they do not retain the full
        # all-layer routed-experts tensor once per pipeline stage.
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    @staticmethod
    def assert_all_consumed() -> None:
        """Check the training forward replayed every recorded route once, and backward replayed
        it once more exactly for the routers that activation recompute re-runs."""
        errors = []
        for replay_idx, replay in enumerate(RoutingReplay.all_routing_replays):
            expected = len(replay.top_indices_list)
            expected_backward = expected if replay.recomputed else 0
            if replay.forward_index != expected or replay.backward_index != expected_backward:
                errors.append(
                    f"router={replay_idx} (layer {replay.layer_id}, recomputed={replay.recomputed}): "
                    f"forward={replay.forward_index}, backward={replay.backward_index}, recorded={expected}"
                )
        if errors:
            raise RuntimeError("R3 routing replay was not consumed as expected: " + "; ".join(errors[:16]))

    def count_route_mismatch(self, natural_indices, replayed_indices, real_rows):
        """Count real rows whose replayed expert set differs from the router's own top-k."""
        assert (
            real_rows.shape[0] == replayed_indices.shape[0]
        ), f"real-row mask has {real_rows.shape[0]} rows but the replay has {replayed_indices.shape[0]}"
        natural_sets = natural_indices.sort(dim=1).values
        replayed_sets = replayed_indices.to(natural_indices.dtype).sort(dim=1).values
        differs = (natural_sets != replayed_sets).any(dim=1)
        counts = torch.stack([real_rows.sum(), (differs & real_rows).sum()])
        self.route_mismatch = counts if self.route_mismatch is None else self.route_mismatch + counts

    @staticmethod
    def reduce_route_mismatch(num_layers: int) -> torch.Tensor:
        """Sum [routers, real tokens, mismatched tokens] per global layer over all ranks.

        A collective that every rank must call; it resets the local counts.
        """
        counts = torch.zeros((num_layers, 3), dtype=torch.int64, device=accelerator.current_device())
        for replay in RoutingReplay.all_routing_replays:
            counts[replay.layer_id, 0] += 1
            if replay.route_mismatch is not None:
                counts[replay.layer_id, 1:] += replay.route_mismatch
                replay.route_mismatch = None
        dist.all_reduce(counts)
        return counts

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        if hasattr(top_indices, "materialize_for_routing_replay"):
            return top_indices.materialize_for_routing_replay("forward")
        return top_indices.to(
            accelerator.current_device(),
            dtype=torch.int32,
            non_blocking=top_indices.is_pinned(),
        )

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        if hasattr(top_indices, "materialize_for_routing_replay"):
            return top_indices.materialize_for_routing_replay("backward")
        return top_indices.to(
            accelerator.current_device(),
            dtype=torch.int32,
            non_blocking=top_indices.is_pinned(),
        )

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        self.route_mismatch = None

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()
        for resource in RoutingReplay.lazy_resources:
            resource.close()
        RoutingReplay.lazy_resources = []

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()

    @staticmethod
    def register_lazy_resource(resource):
        RoutingReplay.lazy_resources.append(resource)

    @staticmethod
    def begin_lazy_pass(release_stage):
        for resource in RoutingReplay.lazy_resources:
            resource.begin_pass(release_stage)


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                probs, top_indices = _compute_topk_for_current_router(
                    old_compute_topk,
                    scores,
                    topk,
                    num_groups=num_groups,
                    group_topk=group_topk,
                )
            elif routing_replay_stage == "record":
                probs, top_indices = _compute_topk_for_current_router(
                    old_compute_topk,
                    scores,
                    topk,
                    num_groups=num_groups,
                    group_topk=group_topk,
                )
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
                if ROUTE_MISMATCH_ROWS is not None:
                    # The router's own pick; detached scores keep it out of the autograd graph.
                    _, natural_indices = _compute_topk_for_current_router(
                        old_compute_topk,
                        scores.detach(),
                        topk,
                        num_groups=num_groups,
                        group_topk=group_topk,
                    )
                    ROUTING_REPLAY.count_route_mismatch(natural_indices, top_indices, ROUTE_MISMATCH_ROWS)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
        else:
            probs, top_indices = _compute_topk_for_current_router(
                old_compute_topk,
                scores,
                topk,
                num_groups=num_groups,
                group_topk=group_topk,
            )

        _capture_ordered_topk(top_indices)
        return probs, top_indices

    return compute_topk


def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1" and not hasattr(module, "routing_replay"):
        module.routing_replay = RoutingReplay()
        _install_router_replay_bridge(module)

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)


class _SlimeRouterReplay:
    def __init__(self, original):
        self.original = original

    def get_replay_topk(self, scores, topk, num_groups, group_topk, compute_topk):
        if self.original is not None:
            original_compute_topk = compute_topk

            def compute_topk(scores, topk, num_groups=None, group_topk=None):
                return self.original.get_replay_topk(scores, topk, num_groups, group_topk, original_compute_topk)

        return get_routing_replay_compute_topk(compute_topk)(scores, topk, num_groups, group_topk)


def _install_router_replay_bridge(module):
    if not hasattr(module, "router_replay") or isinstance(module.router_replay, _SlimeRouterReplay):
        return

    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

    # Older patched Megatron calls the slime hook directly; avoid recording twice.
    if "get_routing_replay_compute_topk" in topk_routing_with_score_function.__code__.co_names:
        return
    module.router_replay = _SlimeRouterReplay(module.router_replay)
