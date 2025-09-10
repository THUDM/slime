from contextlib import nullcontext
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch_memory_saver import torch_memory_saver
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model import get_model_config_from_hf

from slime.backends.utils.data import process_rollout_data
from slime.ray.train_actor import TrainRayActor
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.ppo_utils import compute_approx_kl

from .model import clip_grad_norm, gather_logprobs, train_step
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
        self.model.from_hf(args.load, strict=True)

        self.optim_cfg = AdamWConfig(
            lr=1e-6,
            betas=(0.9, 0.98),
            weight_decay=0.1,
            eps=1e-8,
            foreach=False if args.optimizer_disable_foreach else None,
        )
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
        rollout_data["loss_masks"] = [
            torch.tensor([0] * (len(t) - len(l)) + l, dtype=torch.int, device=torch.cuda.current_device()).unsqueeze(0)
            for t, l in zip(rollout_data["tokens"], rollout_data["loss_masks"])
        ]
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()).unsqueeze(0)
            for t in rollout_data["tokens"]
        ]
        rollout_data["shifted_labels"] = [
            torch.where(l.bool(), t, -100).roll(-1) for t, l in zip(rollout_data["tokens"], rollout_data["loss_masks"])
        ]

        # pack data
        buffer_size = 0
        pack_infos = [[]]
        for i, tokens in enumerate(rollout_data["tokens"]):
            num_token = tokens.numel()
            if num_token + buffer_size > self.args.max_tokens_per_gpu and len(pack_infos[-1]) > 0:
                pack_infos.append([i])
                buffer_size = 0

            pack_infos[-1].append(i)
            buffer_size += num_token

        seq_ctx_list = []
        shifted_labels_list = []
        advantages_list = []
        for indices in pack_infos:
            seq_ctx = [rollout_data["tokens"][i] for i in indices]
            total_len = sum([t.numel() for t in seq_ctx])
            pad_len = self.args.max_tokens_per_gpu - total_len
            label = [rollout_data["shifted_labels"][i] for i in indices]
            advantages = [rollout_data["rewards"][i] for i in indices]
            if pad_len > 0:
                pad_labels = torch.full(
                    (1, pad_len),
                    -100,
                    dtype=rollout_data["shifted_labels"][0].dtype,
                    device=torch.cuda.current_device(),
                )
                seq_ctx.append(
                    torch.zeros(1, pad_len, dtype=rollout_data["tokens"][0].dtype, device=torch.cuda.current_device())
                )
                label.append(pad_labels)
                advantages.append(0.0)

            seq_ctx = SequenceContext.from_input_ids(seq_ctx)
            seq_ctx.num_padding = pad_len
            shifted_labels = torch.cat(label, dim=1)
            advantages = torch.tensor(advantages, device=torch.cuda.current_device()).float().unsqueeze(0)
            cu_seq_lens_q = seq_ctx.cu_seq_lens_q
            num_tokens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
            advantages = torch.repeat_interleave(advantages, num_tokens, dim=1)

            seq_ctx_list.append(seq_ctx)
            shifted_labels_list.append(shifted_labels)
            advantages_list.append(advantages)
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
        sum_old_logprobs: torch.Tensor | None = None
        for old_logprobs, mask in zip(old_logprobs_list, masks):
            old_logprobs = -(old_logprobs * mask).sum()
            sum_old_logprobs = old_logprobs if sum_old_logprobs is None else sum_old_logprobs + old_logprobs
        dist.all_reduce(sum_old_logprobs, op=dist.ReduceOp.SUM)
        avg_logprobs = sum_old_logprobs / global_grad_tokens if global_grad_tokens > 0 else 0

        if dist.get_rank() == 0:
            print(f"Rollout {rollout_id}: avg generation entropy: {avg_logprobs:.4f}")

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
            if dist.get_rank() == 0:
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
            if dist.get_rank() == 0:
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
