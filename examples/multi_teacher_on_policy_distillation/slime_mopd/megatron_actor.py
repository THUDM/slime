"""Megatron teacher backend for teacher-routed MOPD.

Inherits slime's ``MegatronTrainRayActor`` and overrides only ``init``,
``train_actor`` and the rollout-data postprocess hook. slime is not modified.

Teacher lifecycle: every active Megatron teacher is loaded once at init into
slime's pinned-CPU ``TensorBackuper`` under a named ``teacher:<id>`` tag, so each
rollout only pays a CPU->GPU switch instead of re-reading checkpoints. For every
train step the actor partitions the rank-local batch by routed teacher, forwards
only that teacher's samples, and scatters response-aligned log-probs back into
the per-sample ``teacher_log_probs`` list that slime's OPD loss already consumes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from megatron.core import mpu
from slime_mopd.config import MopdConfigError, load_mopd_config
from slime_mopd.megatron_subset import build_teacher_subset, scatter_subset_into
from slime_mopd.routing import TrainSampleView, pop_default_fallback_count, resolve_teacher_id

from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.observability.timer import Timer, inverse_timer, timer

logger = logging.getLogger(__name__)

TEACHER_TAG_PREFIX = "teacher:"


def mopd_megatron_enabled(args: Any) -> bool:
    """True when this run should use the MOPD Megatron teacher path."""

    if not getattr(args, "use_opd", False) or getattr(args, "opd_type", None) != "megatron":
        return False
    try:
        config = load_mopd_config(args)
    except MopdConfigError:
        return False
    return any(teacher.backend == "megatron" for teacher in config.active_teachers())


def _metric_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return component or "unnamed"


class MopdMegatronTrainRayActor(MegatronTrainRayActor):
    """Megatron actor scoring each sample with its routed teacher."""

    def init(self, args, role: str, with_ref: bool = False, with_opd_teacher: bool = False):
        # slime passes with_opd_teacher for the single-teacher ``--opd-teacher-load``
        # path. MOPD loads every teacher from its own config, so suppress it here
        # and register the named teacher backups below instead.
        start_rollout_id = super().init(args, role, with_ref=with_ref, with_opd_teacher=False)

        if args.debug_rollout_only or role != "actor" or not mopd_megatron_enabled(args):
            return start_rollout_id

        self._base_rollout_data_postprocess = self.rollout_data_postprocess
        self.rollout_data_postprocess = self._postprocess_mopd_metrics
        self._active_mopd_teacher_ids: list[str] | None = None

        config = load_mopd_config(args)
        self.mopd_teachers = {
            teacher.id: teacher for teacher in config.active_teachers() if teacher.backend == "megatron"
        }
        self.mopd_teacher_tags: dict[str, str] = {}
        self.mopd_teacher_metric_components: dict[str, str] = {}
        for teacher_id in self.mopd_teachers:
            component = _metric_component(teacher_id)
            if component in self.mopd_teacher_metric_components.values():
                raise ValueError(
                    f"mopd: teacher ids must have distinct metric-safe names; {teacher_id!r} collides at {component!r}"
                )
            self.mopd_teacher_metric_components[teacher_id] = component

        # Base init defers train_wait until it returns; teacher loading is
        # initialization, so fence it out of the wait timer.
        Timer().end("train_wait")
        try:
            if self.args.offload_train:
                self.wake_up()
            for teacher_id, teacher in self.mopd_teachers.items():
                if not teacher.load:
                    raise ValueError(f"mopd: megatron teacher {teacher_id!r} has no 'load' checkpoint path")
                tag = f"{TEACHER_TAG_PREFIX}{teacher_id}"
                self._load_mopd_teacher(tag, teacher)
                self.mopd_teacher_tags[teacher_id] = tag
            self._switch_model("actor")
        finally:
            if self.args.offload_train:
                self.sleep()
            Timer().start("train_wait")

        return start_rollout_id

    def _load_mopd_teacher(self, tag: str, teacher) -> None:
        # Stock load_other_checkpoint only honors a per-step override for its
        # literal "teacher"/"ref" tags; temporarily set ckpt_step for named tags.
        old_step = self.args.ckpt_step
        try:
            self.args.ckpt_step = teacher.ckpt_step
            super().load_other_checkpoint(tag, teacher.load)
        finally:
            self.args.ckpt_step = old_step

    def train_actor(self, rollout_id: int, rollout_data: dict[str, Any], external_data=None) -> None:
        with inverse_timer("train_wait"), timer("mopd_teacher_score"):
            self._score_routed_teachers(rollout_data)
        try:
            return super().train_actor(rollout_id, rollout_data, external_data=external_data)
        finally:
            self._active_mopd_teacher_ids = None

    def _resolve_batch_teacher_ids(self, rollout_data: dict[str, Any]) -> list[str]:
        # When the custom convert hook (slime_mopd.train_data) is
        # enabled, ``source_names`` already carries the per-record teacher id
        # resolved at rollout time; resolve_teacher_id then matches it via
        # has_teacher and the train-side router is never consulted. Without the
        # hook, routing is reconstructed from source_names/sample_indices/
        # default_teacher.
        config = load_mopd_config(self.args)
        sample_count = len(rollout_data["response_lengths"])
        prompts = rollout_data.get("prompt") or [""] * sample_count
        sources = rollout_data.get("source_names") or ["unknown"] * sample_count
        indices = rollout_data.get("sample_indices") or [None] * sample_count

        teacher_ids = []
        for prompt, source, index in zip(prompts, sources, indices, strict=True):
            view = TrainSampleView(prompt=prompt, source_name=source, index=index)
            teacher_ids.append(resolve_teacher_id(config, self.args, view))

        fallback = pop_default_fallback_count()
        if fallback:
            logger.warning(
                "mopd: %d/%d samples had no teacher annotation/dataset binding and used default_teacher %r",
                fallback,
                sample_count,
                config.default_teacher,
            )
        return teacher_ids

    def _score_routed_teachers(self, rollout_data: dict[str, Any]) -> None:
        from slime.backends.megatron_utils.data import get_data_iterator

        teacher_ids = self._resolve_batch_teacher_ids(rollout_data)
        if len(teacher_ids) != len(rollout_data["tokens"]):
            raise ValueError("mopd: teacher route count does not match sample count")

        routed_values: list[Any | None] = [None] * len(teacher_ids)
        produced_values = False
        try:
            for teacher_id in self.mopd_teachers:
                subset, selected = build_teacher_subset(rollout_data, teacher_ids, teacher_id)
                if subset is None:
                    logger.info(
                        "mopd: skip teacher %s (%d/%d samples routed)",
                        teacher_id,
                        sum(1 for routed in teacher_ids if routed == teacher_id),
                        len(teacher_ids),
                    )
                    continue

                self._switch_model(self.mopd_teacher_tags[teacher_id])
                output = self.compute_log_prob(
                    get_data_iterator(subset),
                    [len(subset["micro_batch_indices"])],
                    store_prefix="teacher_",
                )
                values = output.get("teacher_log_probs")
                if values is None:
                    # Non-last pipeline stages do not own response log-probs.
                    continue
                if len(values) != len(selected):
                    raise ValueError(
                        f"mopd: teacher {teacher_id!r} returned {len(values)} samples; expected {len(selected)}"
                    )
                produced_values = True
                scatter_subset_into(routed_values, selected, values)
        finally:
            self._switch_model("actor")

        if produced_values:
            missing = [index for index, value in enumerate(routed_values) if value is None]
            if missing:
                raise ValueError(f"mopd: scorer did not produce values for positions {missing}")
            rollout_data["teacher_log_probs"] = routed_values

        self._active_mopd_teacher_ids = teacher_ids

    def _aggregation_mean_contribution(
        self,
        values: list[torch.Tensor],
        selected_indices: list[int],
        rollout_data: dict[str, Any],
    ) -> tuple[float, float]:
        """Distributed sum/count pair using slime's policy-loss aggregation."""

        from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

        rollout_mask_sums = rollout_data.get("rollout_mask_sums")
        if rollout_mask_sums is None:
            raise ValueError("mopd: per-teacher KL metrics require rollout_mask_sums")

        live = [
            index
            for index in selected_indices
            if rollout_data["loss_masks"][index].sum().item() > 0 and rollout_mask_sums[index].item() > 0
        ]
        if not live:
            return 0.0, 0.0

        reducer = get_sum_of_sample_mean(
            [rollout_data["total_lengths"][index] for index in live],
            [rollout_data["response_lengths"][index] for index in live],
            [rollout_data["loss_masks"][index] for index in live],
            sample_denoms=rollout_mask_sums[live],
        )
        local_sum = reducer(torch.cat([values[index] for index in live], dim=0)).item()
        local_count = (
            sum(rollout_data["loss_masks"][index].sum().item() / rollout_mask_sums[index].item() for index in live)
            / mpu.get_context_parallel_world_size()
        )
        return local_sum, local_count

    def _postprocess_mopd_metrics(self, args: Any, rollout_id: int, rollout_data: dict[str, Any]) -> None:
        from slime.observability.train_metric_utils import gather_log_data

        teacher_ids = self._active_mopd_teacher_ids
        on_metric_rank = mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage()
        if not teacher_ids or not on_metric_rank:
            if self._base_rollout_data_postprocess is not None:
                self._base_rollout_data_postprocess(args, rollout_id, rollout_data)
            return

        reverse_kls = rollout_data.get("opd_reverse_kl")
        if reverse_kls is None or len(reverse_kls) != len(teacher_ids):
            raise ValueError("mopd: opd_reverse_kl must contain one tensor per routed sample")

        log_dict: dict[str, float | tuple[float, float]] = {}
        for teacher_id, component in self.mopd_teacher_metric_components.items():
            selected = [index for index, routed in enumerate(teacher_ids) if routed == teacher_id]
            key = f"mopd/opd_reverse_kl_train_aggregation_mean/{component}"
            if not selected:
                # Emit a zero (sum, count) pair so every metric rank ships the
                # same key set. gather_and_reduce_log_dict indexes all gathered
                # ranks by the source rank's keys, so omitting a key here would
                # either drop the teacher's metric or raise KeyError.
                log_dict[key] = (0.0, 0.0)
                continue
            log_dict[key] = self._aggregation_mean_contribution(reverse_kls, selected, rollout_data)
        gather_log_data("rollout", args, rollout_id, log_dict)

        if self._base_rollout_data_postprocess is not None:
            self._base_rollout_data_postprocess(args, rollout_id, rollout_data)
