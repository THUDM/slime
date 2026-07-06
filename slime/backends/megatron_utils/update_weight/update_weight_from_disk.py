from __future__ import annotations

import shutil
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
import torch.distributed as dist
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from ..hf_checkpoint_saver import save_hf_model_to_path


class UpdateWeightFromDisk:
    """Full-weight sync through a shared filesystem and SGLang disk reload."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self.rollout_engines: Sequence[ActorHandle] = []
        self.rollout_engine_lock: ActorHandle | None = None
        # Post-write hook: object-store-backed shared filesystems lack cross-host
        # read-after-write consistency, so written files need an explicit step
        # (e.g. uploading them to the backing object store) before the engines can see them.
        self._post_write_hook: Callable | None = None
        if args.custom_update_weight_post_write_path:
            from slime.utils.misc import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

    def disconnect_rollout_engines(self) -> None:
        return

    def pop_metrics(self) -> dict[str, float]:
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        version_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{self.weight_version:06d}"

        if dist.get_rank() == 0:
            shutil.rmtree(version_dir, ignore_errors=True)
        dist.barrier(group=get_gloo_group())

        # every writing rank creates the dir itself: a non-POSIX shared filesystem may not surface
        # one rank's mkdir to another until commit
        version_dir.mkdir(parents=True, exist_ok=True)
        save_hf_model_to_path(
            self.args,
            version_dir,
            self.model,
            model_name=self.model_name,
            quantization_config=self.quantization_config,
            progress_desc="Save HF  weights for update from disk",
        )
        dist.barrier(group=get_gloo_group())

        # every rank runs the hook (it gates itself): each container must publish
        # its own writes
        if self._post_write_hook is not None:
            self._post_write_hook(self.args, str(version_dir), list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())

        # SGLang reload is orchestrated by RayTrainGroup after the checkpoint
        # is fully written, so training-side lifecycle can decide whether
        # Megatron actors are still alive.
