from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
from torch.utils._foreach_utils import _device_has_foreach_support, _has_foreach_support
from xtuner.v1.module.router import NoAuxRouterConfig

from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss


def train_step(args, model, model_cfg, optimizer, data_batches: list[dict], global_grad_tokens):
    moe_need_update_bias = (
        isinstance(getattr(model_cfg, "router", None), NoAuxRouterConfig)
        and model_cfg.router.router_bias_update_speed > 0
    )

    if moe_need_update_bias:
        tokens_per_expert_global_for_bias = torch.zeros(
            model_cfg.num_hidden_layers - model_cfg.first_k_dense_replace,
            model_cfg.n_routed_experts,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )

    loss_dict = {}

    def update_loss_dict(key: str, val):
        # TODO: this is slow
        val = val.item() if isinstance(val, torch.Tensor) else val
        if not key.startswith("train/"):
            key = f"train/{key}"
        if key not in loss_dict:
            loss_dict[key] = val
        else:
            loss_dict[key] += val

    for data_batch in data_batches:
        seq_ctx = data_batch["seq_ctx"]
        shifted_labels = data_batch["shifted_labels"]
        old_logprobs = data_batch["old_logprobs"]
        advantages = data_batch["advantages"]
        mask = data_batch["mask"]

        # TODO: check what xtuner does with intra_layer_micro_batch > 1
        output = model(seq_ctx=seq_ctx, loss_ctx=None)

        # llm loss has been global averaged
        logits = output["logits"]
        logprobs = gather_logprobs(logits, shifted_labels)
        ppo_kl = logprobs - old_logprobs

        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
        pg_clipfrac = (pg_clipfrac * mask).sum() / global_grad_tokens
        pg_loss = (pg_loss * mask).sum() / global_grad_tokens
        loss = pg_loss

        if args.use_kl_loss:
            ref_log_probs = data_batch["ref_logprobs"]
            kl = compute_approx_kl(
                logprobs,
                ref_log_probs,
                kl_loss_type=args.kl_loss_type,
            )
            kl_loss = (kl * mask).sum() / global_grad_tokens
            loss = loss + args.kl_loss_coef * kl_loss

        # TODO: find out how to to per token loss and per sample loss for balancing loss and z_loss
        if "balancing_loss" in output:
            balancing_loss = output["balancing_loss"] / len(data_batches)
            loss = loss + balancing_loss
        if "z_loss" in output:
            z_loss = output["z_loss"] / len(data_batches)
            loss = loss + z_loss

        if moe_need_update_bias:
            assert "tokens_per_expert_global" in output, "tokens_per_expert_global is required for bias update."
            tokens_per_expert_global_for_bias += output["tokens_per_expert_global"]

        # update log
        update_loss_dict("loss", loss)
        update_loss_dict("pg_loss", pg_loss)
        update_loss_dict("pg_clipfrac", pg_clipfrac)
        if args.use_kl_loss:
            update_loss_dict("kl_loss", kl_loss)
        if "balancing_loss" in output:
            update_loss_dict("balancing_loss", balancing_loss)
        if "z_loss" in output:
            update_loss_dict("z_loss", z_loss)

        del output

        # we already divide by the global token, need to remove the mean in fsdp gradient allreduce.
        loss = loss * dist.get_world_size()
        loss.backward()

    if moe_need_update_bias:
        avg_count_load = tokens_per_expert_global_for_bias.float().mean(1)
        max_load_i, _ = torch.max(tokens_per_expert_global_for_bias, dim=1)
        maxvio_all_layers = (max_load_i - avg_count_load) / avg_count_load
        maxvio = maxvio_all_layers.mean()
        model.update_bias(tokens_per_expert_global_for_bias, avg_count_load)  # type: ignore

    grad_norm = clip_grad_norm(model, args.clip_grad)

    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        optimizer.zero_grad()
    else:
        optimizer.step()
        optimizer.zero_grad()

    # TODO: this is slow
    reduced_loss_dict = [None] * dist.get_world_size()
    dist.all_gather_object(reduced_loss_dict, loss_dict)
    loss_dict = {key: sum([d[key] for d in reduced_loss_dict]) for key in loss_dict.keys()}
    loss_dict["train/grad_norm"] = grad_norm.item()
    if moe_need_update_bias:
        loss_dict["maxvio"] = maxvio.item()

    return loss_dict


def clip_grad_norm(model, max_grad_norm):
    model.scale_and_reduce_grad()
    params = model.trainable_parameters()
    grads = [p.grad for _, p in params if p.grad is not None]
    grouped_grads = group_tensors_by_device_mesh_and_placements(grads)
    total_norms = []
    for grads in grouped_grads.values():
        total_norm = cal_total_norm(grads, foreach=True)
        total_norms.append(total_norm)
    grad_norm = torch.linalg.vector_norm(torch.stack(total_norms), ord=2.0, dtype=torch.float32)
    clip_coef = max_grad_norm / (grad_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grads in grouped_grads.values():
        device = grads[0].device
        if _device_has_foreach_support(device):
            torch._foreach_mul_(grads, clip_coef_clamped.to(device))
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in grads:
                g.mul_(clip_coef_clamped_device)
    return grad_norm


def group_tensors_by_device_mesh_and_placements(tensors: List[torch.Tensor]):
    grouped_tensors: Dict[Tuple[DeviceMesh, Tuple[Placement, ...]], List[torch.Tensor]] = {}
    for tensor in tensors:
        assert isinstance(tensor, DTensor)
        key = (tensor.device_mesh, tensor.placements)
        if key in grouped_tensors:
            grouped_tensors[key].append(tensor)
        else:
            grouped_tensors[key] = [tensor]
    return grouped_tensors


def cal_total_norm(tensors: List[DTensor], foreach: Optional[bool] = None):
    if len(tensors) == 0:
        return torch.tensor(0.0)

    device_mesh: DeviceMesh = tensors[0].device_mesh
    placements = tensors[0].placements
    device = tensors[0].device
    norms: Tuple[DTensor, ...]
    if (foreach is None and _has_foreach_support(tensors, device)) or (  # type: ignore
        foreach and _device_has_foreach_support(device)
    ):
        norms = torch._foreach_norm(tensors, 2)  # type: ignore
    elif foreach:
        raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
    else:
        norms = tuple(torch.linalg.vector_norm(g, 2) for g in tensors)

    local_norm = torch.linalg.vector_norm(torch.stack([norm.to_local() for norm in norms]), 2, dtype=torch.float32)
    local_norm_squared = local_norm**2
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            # When using ep + fsdp, the placement corresponding to fsdp mesh is _StridedShard
            # isinstance(_StridedShard, Shard) is True
            dist.all_reduce(local_norm_squared, group=device_mesh.get_group(i))
        elif isinstance(placement, Replicate):
            pass
        else:
            raise ValueError(f"Unsupported placement type {placement} in clip_grad_norm")
    global_norm = local_norm_squared**0.5
    return global_norm


def gather_logprobs(logits, shifted_labels):
    logprobs = F.log_softmax(logits.float(), dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shifted_labels.clip(min=0).unsqueeze(-1)).squeeze(-1)
    return -logprobs
