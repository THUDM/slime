"""CI utilities for training backend testing."""

import logging
import math
from argparse import Namespace

import torch

logger = logging.getLogger(__name__)


def check_kl(args: Namespace, log_dict: dict[str, float], step_id: int, accumulated_step_id: int) -> None:
    if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
        if args.multi_latent_attention:
            # TODO: mla currently have non-zero kl, need further investigation
            assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
        else:
            assert log_dict["train/ppo_kl"] == 0.0 and log_dict["train/pg_clipfrac"] == 0.0, f"{log_dict=}"
    if accumulated_step_id == 0 and "train/kl_loss" in log_dict and not args.use_rollout_routing_replay:
        assert log_dict["train/kl_loss"] == 0.0, f"{log_dict=}"


def check_grad_norm(
    args: Namespace,
    grad_norm: float,
    rollout_id: int,
    step_id: int,
    role: str = "actor",
    rank: int = 0,
) -> None:

    if rank != 0:
        return

    if args.ci_save_grad_norm is not None:
        ci_save_grad_norm_path = args.ci_save_grad_norm.format(
            role=role,
            rollout_id=rollout_id,
            step_id=step_id,
        )
        torch.save(grad_norm, ci_save_grad_norm_path)

    elif args.ci_load_grad_norm is not None:
        ci_load_grad_norm_path = args.ci_load_grad_norm.format(
            role=role,
            rollout_id=rollout_id,
            step_id=step_id,
        )
        expected_grad_norm = torch.load(ci_load_grad_norm_path, weights_only=False)
        assert math.isclose(
            grad_norm,
            expected_grad_norm,
            rel_tol=0.03,
            abs_tol=0.03,
        ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
