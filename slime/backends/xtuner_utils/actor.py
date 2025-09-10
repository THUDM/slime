from contextlib import nullcontext
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch_memory_saver import torch_memory_saver
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.rl.base import TrainingController as _RayTrainingController
from xtuner.v1.rl.grpo import GRPOLossConfig

from slime.backends.utils.data import process_rollout_data
from slime.ray.train_actor import TrainRayActor
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.ppo_utils import compute_approx_kl

from .model import clip_grad_norm, gather_logprobs, sp_split, train_step
from .update_weight_utils import UpdateWeightFromDistributed


class XTunerTrainRayActor(TrainRayActor):
    def init(self, args, role, wandb_run_id, with_ref: bool = False):
        super().init(args, role, wandb_run_id, with_ref)

        torch.manual_seed(args.seed)

        self.model_cfg = get_model_config_from_hf(args.hf_checkpoint)
        self.fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=args.ep_size)

        if self.with_ref:
            with torch.device("meta"):
                ref_model = self.model_cfg.build()
            self.ref_model = ref_model.fully_shard(self.fsdp_cfg)
            self.ref_model.from_hf(args.ref_load, strict=True)
            self.ref_model.eval()
            self.ref_model.to_device("cpu")

        with torch.device("meta"):
            model = self.model_cfg.build()
        self.model = model.fully_shard(self.fsdp_cfg)
        print(f"load from: {args.load}")
        self.model.from_hf(args.load, strict=True)

        self.optim_cfg = AdamWConfig(lr=1e-6, foreach=False if args.optimizer_disable_foreach else None)
        self.optimizer = self.optim_cfg.build([p for p in self.model.parameters() if p.requires_grad])

        # init sp mesh
        world_size = dist.get_world_size()
        assert world_size % args.sp_size == 0, f"world_size {world_size} must be divisible by sp_size {args.sp_size}"
        dp_size = world_size // args.sp_size

        self.data_mesh = init_device_mesh(
            "cuda" if not self.fsdp_cfg.cpu_offload else "cpu",
            (dp_size, args.sp_size),
            mesh_dim_names=("dp", "sp"),
        )
        self.sp_mesh = self.data_mesh["sp"]

        # loss cfg
        self.loss_cfg = GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=args.eps_clip_high,
                cliprange_low=args.eps_clip,
                loss_type=args.policy_loss_type,
            ),
            ignore_idx=-100,
            use_kl_loss=self.with_ref,
            kl_loss_coef=0.001,
            kl_loss_type="low_var_kl",
            mode="chunk",
            chunk_size=512,
        )

        # only for its utils
        TrainingController = _unwrap_ray_actor(_RayTrainingController)
        self.controller = TrainingController([])
        # borrow utility methods if present
        if hasattr(self.controller, "_packing"):
            self._packing = self.controller._packing

        self.weight_updator = UpdateWeightFromDistributed(args, self.model)

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

        path = f"{self.args.save}/iter_{iteration:07}/hf"
        self.model.save_hf(path, save_dtype=torch.bfloat16)

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        self.weight_updator.connect_rollout_engines(rollout_engines, rollout_engine_lock)
        dist.barrier(group=get_gloo_group())

    def get_rollout_data(self, rollout_data_ref):
        dp_rank = dist.get_rank() // self.args.sp_size
        dp_size = dist.get_world_size() // self.args.sp_size
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
        packed_data_batches = sorted(packed_data_batches, key=lambda x: x["seq_ctx"].max_length_q, reverse=True)

        seq_ctx_list: list[SequenceContext] = []
        shifted_labels_list = []
        advantages_list = []
        for data in packed_data_batches:
            if self.sp_mesh.size() > 1:
                data["seq_ctx"] = data["seq_ctx"].split(self.sp_mesh)
                data["shifted_labels"] = sp_split(
                    data["shifted_labels"], sp_mesh=self.sp_mesh, split_dim=1, padding_value=-100
                )
                data["advantages"] = sp_split(data["advantages"], sp_mesh=self.sp_mesh, split_dim=1, padding_value=0.0)

            seq_ctx_list.append(data["seq_ctx"].to(torch.cuda.current_device()))
            shifted_labels_list.append(data["shifted_labels"].to(torch.cuda.current_device()))
            advantages_list.append(
                torch.tensor(data["advantages"], dtype=torch.float, device=torch.cuda.current_device())
            )

        return seq_ctx_list, shifted_labels_list, advantages_list

    def compute_logprobs(
        self, model, seq_ctx_list: list[SequenceContext], shifted_labels_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        logprobs_list = []
        with torch.no_grad():
            for seq_ctx, labels in zip(seq_ctx_list, shifted_labels_list):
                output = model(seq_ctx=seq_ctx, loss_ctx=None)
                logprobs = gather_logprobs(output["logits"], labels)
                logprobs_list.append(logprobs)
                del output
        return logprobs_list

    def train(self, rollout_id, rollout_data_ref):  # type: ignore[override]
        if self.args.offload:
            self.wake_up(("model"))

        seq_ctx_list, shifted_labels_list, advantages_list = self.get_rollout_data(rollout_data_ref)
        masks = [labels != -100 for labels in shifted_labels_list]

        rank_grad_tokens: torch.Tensor | None = None
        for mask in masks:
            grad_tokens = mask.sum()
            rank_grad_tokens = grad_tokens if rank_grad_tokens is None else rank_grad_tokens + grad_tokens
        rank_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        global_grad_tokens = rank_grad_tokens
        dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)

        # old logprobs are inplaced updated in compute_actor_logprobs
        old_logprobs_list = self.compute_logprobs(self.model, seq_ctx_list, shifted_labels_list)
        sum_entropy: torch.Tensor | None = None
        for old_logprobs, mask in zip(old_logprobs_list, masks):
            entropy = -(old_logprobs * mask).sum()
            sum_entropy = entropy if sum_entropy is None else sum_entropy + entropy
        dist.all_reduce(sum_entropy, op=dist.ReduceOp.SUM)
        avg_gen_entropy = sum_entropy / global_grad_tokens if global_grad_tokens > 0 else 0
        print(f"Rollout {rollout_id}: avg generation entropy: {avg_gen_entropy:.4f}")

        if self.with_ref:
            self.ref_model.to_device(torch.cuda.current_device())
            ref_logprobs_list = self.compute_logprobs(seq_ctx_list, shifted_labels_list)
            self.ref_model.to_device("cpu")

            kl_div_sum: torch.Tensor | None = None
            for old_logprobs, ref_logprobs, mask in zip(old_logprobs_list, ref_logprobs_list, masks):
                kl_div = (
                    compute_approx_kl(old_logprobs, ref_logprobs, loss_weights=mask, kl_loss_type="low_var_kl")
                    * (mask.to(old_logprobs.dtype))
                ).sum()

                kl_div_sum = kl_div if kl_div_sum is None else kl_div_sum + kl_div

            kl_div_sum = cast(torch.Tensor, kl_div_sum)
            dist.all_reduce(kl_div_sum, op=dist.ReduceOp.SUM)
            avg_kl_div = kl_div_sum / global_grad_tokens if global_grad_tokens > 0 else 0
            print(f"Rollout {rollout_id}: avg KL divergence: {avg_kl_div:.4f}")

        iters_per_step = len(seq_ctx_list) // 1
        for i in range(0, len(seq_ctx_list), iters_per_step):
            loss_log, other_log = train_step(
                self.args,
                self.model,
                self.model_cfg,
                data_batches=[
                    {
                        "seq_ctx": seq_ctx_list[i + j],
                        "shifted_labels": shifted_labels_list[i + j],
                        "advantages": advantages_list[i + j],
                        "old_logprobs": old_logprobs_list[i + j],
                        "ref_logprobs": ref_logprobs_list[i + j] if self.with_ref else None,
                        "mask": masks[i + j],
                    }
                    for j in range(iters_per_step)
                ],
            )
            grad_norm = clip_grad_norm(self.model, self.args.clip_grad)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()

            log_info = dict()
            log_info.update(loss_log)
            log_info.update(other_log)
            log_info["grad_norm"] = grad_norm.item()
            log_str = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in log_info.items()
            )
            log_str = f"Rollout {rollout_id} Step {i}: " + log_str
            print(log_str)

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


# Unwrap Ray actor to get the original (non-remote) TrainingController class
def _unwrap_ray_actor(actor_cls):
    # Ray >= 2.x exposes metadata with the original class
    orig = getattr(actor_cls, "__ray_metadata__", None)
    if orig is not None and getattr(orig, "cls", None) is not None:
        return orig.cls
    # Older Ray versions sometimes expose __ray_actor_class__
    return getattr(actor_cls, "__ray_actor_class__", actor_cls)
