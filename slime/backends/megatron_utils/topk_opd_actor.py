import logging
import math
import os

import torch
from slime.utils import train_dump_utils
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import inverse_timer, timer
from slime.utils.types import RolloutBatch

from .actor import MegatronTrainRayActor
from .data import get_data_iterator, log_perf_data, log_rollout_data
from .initialize import is_megatron_main_rank
from .loss import get_topk_log_probs
from .model import forward_only, train

logger = logging.getLogger(__name__)


class TopKOPDMegatronTrainRayActor(MegatronTrainRayActor):
    """Megatron actor variant for top-k level OPD.

    The base actor keeps token-level OPD integrated through advantages. This
    subclass prepares top-k/tail distributions from old actor and one or more
    homogeneous teachers, then trains with ``topk_opd_loss``.
    """

    def _teacher_tags(self) -> list[str]:
        tags = [tag for tag in self.weights_backuper.backup_tags if tag == "teacher" or tag.startswith("teacher_")]
        return sorted(tags, key=lambda tag: (-1 if tag == "teacher" else int(tag.rsplit("_", 1)[1])))

    def compute_topk_log_prob(
        self,
        data_iterator,
        num_microbatches,
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        with timer(f"{store_prefix}topk_log_probs"):
            return forward_only(
                get_topk_log_probs,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                extra_kwargs={"top_k": self.args.opd_top_k},
            )

    def _compute_old_topk_data(self, data_iterator, num_microbatches, rollout_data: RolloutBatch) -> None:
        self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
        if self.args.use_routing_replay:
            if self.args.use_rollout_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
            else:
                os.environ["ROUTING_REPLAY_STAGE"] = "record"
        topk_data = self.compute_topk_log_prob(data_iterator, num_microbatches)
        rollout_data["student_topk_indices"] = topk_data["topk_indices"]
        rollout_data["old_topk_log_probs"] = topk_data["topk_log_probs"]
        rollout_data["old_tail_log_probs"] = topk_data["tail_log_probs"]
        if self.args.use_rollout_routing_replay:
            RoutingReplay.clear_all_forward()

    def _compute_teacher_topk_data(self, data_iterator, num_microbatches, rollout_data: RolloutBatch) -> None:
        teacher_tags = self._teacher_tags()
        if not teacher_tags:
            raise ValueError("--topk-level-opd requires at least one Megatron teacher checkpoint.")

        teacher_outputs = []
        for teacher_tag in teacher_tags:
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
            self._switch_model(teacher_tag)
            teacher_outputs.append(
                self.compute_topk_log_prob(data_iterator, num_microbatches, store_prefix=f"{teacher_tag}_")
            )

        topk_keys = [f"{teacher_tag}_topk_log_probs" for teacher_tag in teacher_tags]
        tail_keys = [f"{teacher_tag}_tail_log_probs" for teacher_tag in teacher_tags]
        n_teachers = len(teacher_tags)

        rollout_data["teacher_topk_log_probs"] = [
            torch.logsumexp(
                torch.stack([out[key][sample_idx] for out, key in zip(teacher_outputs, topk_keys, strict=True)]),
                dim=0,
            )
            - math.log(n_teachers)
            for sample_idx in range(len(teacher_outputs[0][topk_keys[0]]))
        ]
        rollout_data["teacher_tail_log_probs"] = [
            torch.logsumexp(
                torch.stack([out[key][sample_idx] for out, key in zip(teacher_outputs, tail_keys, strict=True)]),
                dim=0,
            )
            - math.log(n_teachers)
            for sample_idx in range(len(teacher_outputs[0][tail_keys[0]]))
        ]

    def _prepare_topk_opd_data(self, data_iterator, num_microbatches, rollout_data: RolloutBatch) -> None:
        self._compute_old_topk_data(data_iterator, num_microbatches, rollout_data)
        self._compute_teacher_topk_data(data_iterator, num_microbatches, rollout_data)
        if self._active_model_tag != "actor":
            self._switch_model("actor")

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> None:
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            self._prepare_topk_opd_data(data_iterator, num_microbatches, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            log_rollout_data(rollout_id, self.args, rollout_data)

            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    global_batch_sizes,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        self.weights_backuper.backup("actor")

        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())
