from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch

from slime.data.tensor import TensorRef
from slime.utils import accelerator

logger = logging.getLogger(__name__)


def validate_routed_experts_tensor(
    experts: torch.Tensor,
    args,
    *,
    sample_index: int | None = None,
    expected_rows: int | None = None,
) -> None:
    """Validate one rollout route tensor before publication or R3 consumption."""

    num_layers = int(args.num_layers)
    topk = int(args.moe_router_topk)
    if experts.ndim != 3 or tuple(experts.shape[1:]) != (num_layers, topk):
        raise ValueError(
            "Invalid rollout routed-experts shape for R3: "
            f"sample={sample_index}, got={tuple(experts.shape)}, expected=(*, {num_layers}, {topk})."
        )
    if experts.shape[0] == 0:
        raise ValueError(f"R3 sample {sample_index} has no routed-experts rows.")
    if expected_rows is not None and experts.shape[0] != expected_rows:
        raise ValueError(
            f"R3 sample {sample_index} routed-experts rows={experts.shape[0]}, "
            f"expected={expected_rows} from len(tokens)-1."
        )

    moe_layer_freq = getattr(args, "moe_layer_freq", None)
    if isinstance(moe_layer_freq, (list, tuple)):
        moe_layers = [layer_id for layer_id, freq in enumerate(moe_layer_freq[:num_layers]) if int(freq) != 0]
    elif isinstance(moe_layer_freq, int) and moe_layer_freq > 0:
        moe_layers = [layer_id for layer_id in range(num_layers) if layer_id % moe_layer_freq == 0]
    else:
        moe_layers = list(range(num_layers))
    if topk > 1:
        moe_routes = experts[:, moe_layers, :]
        token_layer_present = torch.any(moe_routes != 0, dim=-1)
        missing_layer_mask = ~torch.any(token_layer_present, dim=0)
        missing_layers = [moe_layers[idx] for idx in torch.nonzero(missing_layer_mask).reshape(-1).tolist()]
        if missing_layers:
            raise ValueError(
                "R3 routed-experts capture is all zero for MoE layers "
                f"{missing_layers} in sample {sample_index}. This usually means SGLang pipeline stages "
                "did not aggregate their disjoint routing captures; refusing to replay expert 0 everywhere."
            )
        missing_token_layers = ~token_layer_present
        if torch.any(missing_token_layers):
            # Partial layers indicate broken PP aggregation. Fully missing rows
            # are known request-level capture holes and replay expert 0.
            fully_missing_rows = torch.all(missing_token_layers, dim=1)
            partial_missing = missing_token_layers & ~fully_missing_rows.unsqueeze(1)
            if torch.any(partial_missing):
                missing = torch.nonzero(partial_missing, as_tuple=False)
                examples = [
                    (int(token_idx), int(moe_layers[int(moe_idx)])) for token_idx, moe_idx in missing[:16].tolist()
                ]
                raise ValueError(
                    "R3 routed-experts capture is all zero for token/layer pairs "
                    f"{examples} in sample {sample_index}. This usually means prompt/cache/PD "
                    "routes were only partially returned."
                )
            num_hole_rows = int(fully_missing_rows.sum())
            allowed_hole_rows = max(64, experts.shape[0] // 100)
            if num_hole_rows > allowed_hole_rows:
                hole_positions = torch.nonzero(fully_missing_rows).reshape(-1)[:16].tolist()
                raise ValueError(
                    f"R3 sample {sample_index} has {num_hole_rows}/{experts.shape[0]} rows with no "
                    f"routing capture at all (e.g. token rows {hole_positions}), exceeding the "
                    f"tolerated {allowed_hole_rows}. This looks like a systematic capture failure, "
                    "not isolated prefix-cache/PD holes."
                )
            logger.warning(
                "R3 sample %s: tolerating %d/%d fully-missing routing rows; these rows replay expert 0.",
                sample_index,
                num_hole_rows,
                experts.shape[0],
            )

    if moe_layers:
        # Cast first so uint8 expert ids compare correctly with num_experts=256.
        moe_routes = experts[:, moe_layers, :].to(torch.int64)
        num_experts = int(getattr(args, "num_experts", torch.iinfo(torch.int32).max))
        invalid = (moe_routes < 0) | (moe_routes >= num_experts)
        if torch.any(invalid):
            invalid_ids = torch.unique(moe_routes[invalid])[:16].tolist()
            raise ValueError(
                f"R3 sample {sample_index} contains routed expert ids outside "
                f"[0, {num_experts - 1}]: {invalid_ids}."
            )


def validate_routed_experts_value(
    value: torch.Tensor | TensorRef,
    args,
    *,
    sample_index: int | None,
    expected_rows: int | None = None,
) -> None:
    if isinstance(value, TensorRef) and value.validated:
        expected_tail = (int(args.num_layers), int(args.moe_router_topk))
        if len(value.shape) != 3 or tuple(value.shape[1:]) != expected_tail or value.shape[0] <= 0:
            raise ValueError(
                "Invalid rollout routed-experts disk reference for R3: "
                f"sample={sample_index}, shape={value.shape}, expected=(*, {expected_tail[0]}, {expected_tail[1]})."
            )
        if expected_rows is not None and value.shape[0] != expected_rows:
            raise ValueError(
                f"R3 sample {sample_index} routed-experts rows={value.shape[0]}, "
                f"expected={expected_rows} from len(tokens)-1."
            )
        return

    tensor = value.load() if isinstance(value, TensorRef) else value
    validate_routed_experts_tensor(tensor, args, sample_index=sample_index, expected_rows=expected_rows)


def materialize_routed_experts(value: Any, *, pin_memory: bool = False) -> torch.Tensor:
    if isinstance(value, TensorRef):
        return value.load(pin_memory=pin_memory)
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


class RoutedExpertsMicrobatchPrefetcher:
    """Bounded disk-to-CPU prefetch queue shared by one actor rollout."""

    def __init__(self, prefetch_microbatches: int) -> None:
        self.prefetch_microbatches = max(0, int(prefetch_microbatches))
        self.sources: list[RoutedExpertsMicrobatch] = []
        # Which stage's completed consumption frees a microbatch. "forward"
        # suits forward-only passes (log probs); the train pass must hold each
        # microbatch until its recompute-backward pops it, or 1F1B re-reads
        # every microbatch from disk between forward and backward.
        self.release_stage = "forward"
        self._executor = (
            ThreadPoolExecutor(
                max_workers=min(2, self.prefetch_microbatches + 1),
                thread_name_prefix="r3-prefetch",
            )
            if self.prefetch_microbatches > 0
            else None
        )
        self._closed = False

    def add(self, source: RoutedExpertsMicrobatch) -> None:
        source.index = len(self.sources)
        source.prefetcher = self
        self.sources.append(source)

    def start(self) -> None:
        for index in range(min(len(self.sources), self.prefetch_microbatches)):
            self.sources[index].prefetch()

    def begin_pass(self, release_stage: str) -> None:
        """Arm the release policy for the next pass and re-warm its first window.

        Sources released by the previous pass would otherwise load synchronously
        on the training thread when the new pass consumes its first microbatch.
        """
        self.release_stage = release_stage
        self.start()

    def prefetch_after(self, index: int) -> None:
        for next_index in range(index + 1, min(len(self.sources), index + 1 + self.prefetch_microbatches)):
            self.sources[next_index].prefetch()

    @property
    def executor(self) -> ThreadPoolExecutor | None:
        return None if self._closed else self._executor

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        for source in self.sources:
            source.release()


class RoutedExpertsMicrobatch:
    """Lazily prepares one training-layout routed-experts microbatch."""

    def __init__(
        self,
        values: list[TensorRef],
        tokens: list[torch.Tensor],
        *,
        consumer_count: int,
        prepare_kwargs: dict[str, Any],
    ) -> None:
        self.values = values
        self.tokens = tokens
        self.consumer_count = consumer_count
        self.prepare_kwargs = prepare_kwargs
        self.index = -1
        self.prefetcher: RoutedExpertsMicrobatchPrefetcher | None = None
        self._future: Future | None = None
        self._cpu_tensor: torch.Tensor | None = None
        self._lock = threading.Lock()
        self._consumed = {"forward": 0, "backward": 0}

    def _load_and_prepare(self) -> torch.Tensor:
        from slime.backends.megatron_utils.cp_utils import prepare_routed_experts_for_routing_replay

        # Read only the CP/TP rows owned by this rank in the layout helper.
        tensors = [
            (value if isinstance(value, TensorRef) else materialize_routed_experts(value)) for value in self.values
        ]
        return prepare_routed_experts_for_routing_replay(
            tensors,
            self.tokens,
            **self.prepare_kwargs,
        )

    def prefetch(self) -> None:
        with self._lock:
            if self._cpu_tensor is not None or self._future is not None:
                return
            executor = self.prefetcher.executor if self.prefetcher is not None else None
            if executor is not None:
                self._future = executor.submit(self._load_and_prepare)

    def _get_cpu_tensor(self) -> torch.Tensor:
        with self._lock:
            tensor = self._cpu_tensor
            future = self._future
        if tensor is None:
            tensor = future.result() if future is not None else self._load_and_prepare()
            with self._lock:
                self._cpu_tensor = tensor
                self._future = None
        if self.prefetcher is not None:
            self.prefetcher.prefetch_after(self.index)
        return tensor

    def layer_to_cuda(self, layer_id: int, stage: str) -> torch.Tensor:
        cpu_tensor = self._get_cpu_tensor()
        result = cpu_tensor[:, layer_id].to(
            accelerator.current_device(),
            dtype=torch.int32,
            non_blocking=cpu_tensor.is_pinned(),
        )
        self._consumed[stage] += 1
        if self._consumed[stage] == self.consumer_count:
            self._consumed[stage] = 0
            release_stage = self.prefetcher.release_stage if self.prefetcher is not None else stage
            if stage == release_stage:
                self.release()
        return result

    def release(self) -> None:
        with self._lock:
            future = self._future
            self._future = None
            self._cpu_tensor = None
        if future is not None and not future.done():
            future.cancel()


class RoutedExpertsLayerRef:
    """One layer view stored in RoutingReplay instead of a resident CPU tensor."""

    def __init__(self, source: RoutedExpertsMicrobatch, layer_id: int) -> None:
        self.source = source
        self.layer_id = layer_id

    def materialize_for_routing_replay(self, stage: str) -> torch.Tensor:
        return self.source.layer_to_cuda(self.layer_id, stage)
