import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch_memory_saver import torch_memory_saver
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig
from xtuner.v1.rl.base import TrainingController as _RayTrainingController
from xtuner.v1.rl.base import TrainingWorker, WorkerConfig
from xtuner.v1.rl.grpo import GRPOLossConfig

from slime.backends.utils.data import process_rollout_data
from slime.ray.train_actor import TrainRayActor
from slime.utils.distributed_utils import get_gloo_group, init_gloo_group

from .update_weight_utils import UpdateWeightFromDistributed


class XTunerTrainRayActor(TrainRayActor):
    def init(self, args, role, wandb_run_id, with_ref: bool = False):
        self.args = args
        torch.manual_seed(args.seed)

        # Unwrap Ray actor to get the original (non-remote) TrainingController class
        def _unwrap_ray_actor(actor_cls):
            # Ray >= 2.x exposes metadata with the original class
            orig = getattr(actor_cls, "__ray_metadata__", None)
            if orig is not None and getattr(orig, "cls", None) is not None:
                return orig.cls
            # Older Ray versions sometimes expose __ray_actor_class__
            return getattr(actor_cls, "__ray_actor_class__", actor_cls)

        self.worker_cfg: WorkerConfig = WorkerConfig(
            model_cfg=Qwen3Dense8BConfig(),
            optim_cfg=AdamWConfig(lr=1e-6, foreach=False if args.optimizer_disable_foreach else None),
            loss_cfg=GRPOLossConfig(
                policy_loss_cfg=dict(
                    cliprange_high=0.2,
                    cliprange_low=0.2,
                    loss_type=args.policy_loss_type,
                ),
                ignore_idx=-100,
                use_kl_loss=True,
                kl_loss_coef=0.001,
                kl_loss_type="low_var_kl",
                mode="chunk",
                chunk_size=512,
            ),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
            fsdp_cfg=FSDPConfig(
                torch_compile=False,
                cpu_offload=False,
                ep_size=1,
            ),
            load_from=args.hf_checkpoint,
            sp_size=1,
            optimizer_steps=args.train_optimizer_steps,
            pack_max_length=args.pack_max_length,
        )

        self.worker = TrainingWorker(
            self.worker_cfg,
            rank=os.environ["RANK"],
            master_addr=os.environ["MASTER_ADDR"],
            master_port=int(os.environ["MASTER_PORT"]),
            world_size=os.environ["WORLD_SIZE"],
        )
        # only for its utils
        TrainingController = _unwrap_ray_actor(_RayTrainingController)
        self.controller = TrainingController([])
        # borrow utility methods if present
        if hasattr(self.controller, "_packing"):
            self._packing = self.controller._packing  # type: ignore[attr-defined]
        if hasattr(self.controller, "_grouped_by_max_length"):
            self._grouped_by_max_length = self.controller._grouped_by_max_length  # type: ignore[attr-defined]
        init_gloo_group()

        self.weight_updator = UpdateWeightFromDistributed(args, self.worker)

    def sleep(self, tags):
        if not getattr(self.args, "offload", False):
            return
        if torch_memory_saver is not None:
            torch_memory_saver.pause()

    def wake_up(self, tags):
        if not getattr(self.args, "offload", False):
            return
        if torch_memory_saver is not None:
            torch_memory_saver.resume()

    def save_model(self, iteration, with_optimizer=True):
        if self.args.debug_rollout_only:
            return

        raise NotImplementedError()

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        self.weight_updator.connect_rollout_engines(rollout_engines, rollout_engine_lock)
        dist.barrier(group=get_gloo_group())

    def get_rollout_data(self, rollout_data_ref):
        dp_rank = dist.get_rank() % self.worker_cfg.sp_size
        dp_size = dist.get_world_size() // self.worker_cfg.sp_size
        rollout_data = process_rollout_data(self.args, rollout_data_ref, dp_rank, dp_size)
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor([0] * (len(t) - len(l)) + l, dtype=torch.int, device=torch.cuda.current_device())
            for t, l in zip(rollout_data["tokens"], rollout_data["loss_masks"])
        ]

        data_batches = []
        for tokens, reward, loss_mask in zip(
            rollout_data["tokens"],
            rollout_data["rewards"],
            rollout_data["loss_masks"],
        ):
            # TODO: set pack max length in xtuner
            data_batches.append(
                dict(
                    seq_ctx=SequenceContext.from_input_ids((tokens.unsqueeze(0),), device="cpu"),
                    shifted_labels=torch.where(loss_mask.bool(), tokens, -100).roll(-1).unsqueeze(0),
                    advantage=reward,
                )
            )

        packed_data_batches = self._packing(data_batches, self.args.pack_max_length)
        packed_data_batches = self._grouped_by_max_length(packed_data_batches)

        return packed_data_batches

    def train(self, rollout_id, rollout_data_ref):  # type: ignore[override]
        if self.args.offload:
            self.wake_up(("model"))

        data_batches = self.get_rollout_data(rollout_data_ref)
        self.worker.fit(data_batches, rollout_id)

        return

    def update_weights(self):  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.offload:
            # TODO: don't wake up here
            self.wake_up(("model"))

        with torch_memory_saver.disable() if self.args.offload and not torch.version.hip else nullcontext():
            self.weight_updator.update_weights()

        if self.args.offload:
            # TODO: don't wake up here
            self.sleep(("model"))
