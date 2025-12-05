# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate scheduler for FSDP training."""

import logging
import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


class FSDPLRScheduler(LRScheduler):
    """Learning rate scheduler for FSDP training.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        init_lr (float): Initial learning rate.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        lr_warmup_steps (int): Number of warmup steps.
        lr_decay_steps (int): Number of decay steps.
        lr_decay_style (str): Decay style for learning rate.
        use_checkpoint_lr_scheduler (bool, optional): Whether to use the checkpoint values
            for the lr scheduler.
        override_lr_scheduler (bool, optional): Whether to override the lr scheduler values
            with the class values.
        wsd_decay_steps (int, optional): Number of weight decay decay steps.
        lr_wsd_decay_style (str, optional): Decay style for learning rate during weight decay decay
            steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        use_checkpoint_lr_scheduler: Optional[bool] = True,
        override_lr_scheduler: Optional[bool] = False,
        wsd_decay_steps: Optional[int] = None,
        lr_wsd_decay_style: Optional[str] = None,
    ) -> None:
        # Class values.
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style

        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style
        if self.lr_decay_style == "WSD":
            assert self.wsd_decay_steps is not None

        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler

        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, (
                "both override and use-checkpoint are set."
            )

        # Set the learning rate
        self.step(0)

        logger.info(f"> learning rate decay style: {self.lr_decay_style}")

    def get_lr(self, param_group: Optional[dict] = None) -> float:
        """Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4

        Args:
            param_group (dict, optional): parameter group from the optimizer.
        """
        max_lr = param_group.get('max_lr', self.max_lr) if param_group else self.max_lr
        min_lr = param_group.get('min_lr', self.min_lr) if param_group else self.min_lr

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return self.init_lr + (
                (max_lr - self.init_lr) * float(self.num_steps) / float(self.lr_warmup_steps)
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == "constant":
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == "inverse-square-root":
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0

        delta_lr = max_lr - min_lr
        coeff = None

        if self.lr_decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.lr_decay_style == "WSD":
            wsd_anneal_start_ = self.lr_decay_steps - self.wsd_decay_steps
            if self.num_steps <= wsd_anneal_start_:
                coeff = 1.0
            else:
                wsd_steps = self.num_steps - wsd_anneal_start_
                wsd_decay_ratio = float(wsd_steps) / float(self.wsd_decay_steps)
                if self.lr_wsd_decay_style == "linear":
                    coeff = 1.0 - wsd_decay_ratio
                elif self.lr_wsd_decay_style == "cosine":
                    coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
                elif self.lr_wsd_decay_style == "exponential":
                    coeff = (2.0 * math.pow(0.5, wsd_decay_ratio)) - 1.0
                elif self.lr_wsd_decay_style == "minus_sqrt":
                    coeff = 1.0 - math.sqrt(wsd_decay_ratio)
        else:
            raise Exception(f"{self.lr_decay_style} decay style is not supported.")

        assert coeff is not None
        return min_lr + coeff * delta_lr

    def step(self, increment: int = 1) -> None:
        """Set lr for all parameters groups.

        Args:
            increment (int): number of steps to increment
        """
        self.num_steps += increment
        current_lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def state_dict(self) -> dict:
        """Return the state dict."""
        state_dict = {
            "max_lr": self.max_lr,
            "lr_warmup_steps": self.lr_warmup_steps,
            "num_steps": self.num_steps,
            "lr_decay_style": self.lr_decay_style,
            "lr_decay_steps": self.lr_decay_steps,
            "min_lr": self.min_lr,
        }
        return state_dict

    def _check_and_set(self, cls_value: float, sd_value: float, name: str) -> float:
        """Auxiliary function for checking the values in the checkpoint and
        setting them.

        Args:
            cls_value (float): class value
            sd_value (float): checkpoint value
            name (str): name of the parameter
        """
        if self.override_lr_scheduler:
            logger.info(f"> overriding {name} value to {cls_value}")
            return cls_value
        if not self.use_checkpoint_lr_scheduler:
            assert cls_value == sd_value, (
                f"FSDPLRScheduler: class input value {cls_value} and checkpoint"
                f"value {sd_value} for {name} do not match"
            )
        logger.info(f"> using checkpoint value {sd_value} for {name}")
        return sd_value

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dict.

        Args:
            state_dict (dict): state dict to be load
        """
        if "start_lr" in state_dict:
            max_lr_ = state_dict["start_lr"]
        else:
            max_lr_ = state_dict["max_lr"]

        self.max_lr = self._check_and_set(self.max_lr, max_lr_, "learning rate")
        self.min_lr = self._check_and_set(
            self.min_lr, state_dict["min_lr"], "minimum learning rate"
        )

        if "warmup_iter" in state_dict:
            lr_warmup_steps_ = state_dict["warmup_iter"]
        elif "warmup_steps" in state_dict:
            lr_warmup_steps_ = state_dict["warmup_steps"]
        else:
            lr_warmup_steps_ = state_dict["lr_warmup_steps"]

        self.lr_warmup_steps = self._check_and_set(
            self.lr_warmup_steps, lr_warmup_steps_, "warmup iterations"
        )

        if "end_iter" in state_dict:
            lr_decay_steps_ = state_dict["end_iter"]
        elif "decay_steps" in state_dict:
            lr_decay_steps_ = state_dict["decay_steps"]
        else:
            lr_decay_steps_ = state_dict["lr_decay_steps"]

        self.lr_decay_steps = self._check_and_set(
            self.lr_decay_steps, lr_decay_steps_, "total number of iterations"
        )

        if "decay_style" in state_dict:
            lr_decay_style_ = state_dict["decay_style"]
        else:
            lr_decay_style_ = state_dict["lr_decay_style"]

        self.lr_decay_style = self._check_and_set(
            self.lr_decay_style, lr_decay_style_, "learning rate decay style"
        )

        if "num_iters" in state_dict:
            num_steps = state_dict["num_iters"]
        else:
            num_steps = state_dict["num_steps"]

        self.step(increment=num_steps)


def get_lr_scheduler(args, optimizer: torch.optim.Optimizer) -> FSDPLRScheduler:
    """Create and configure the learning-rate scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args: Training/runtime arguments (namespace).
        optimizer (torch.optim.Optimizer): Optimizer bound to the model.

    Returns:
        FSDPLRScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training.
    if hasattr(args, 'train_iters') and hasattr(args, 'global_batch_size'):
        train_iters = args.train_iters
        global_batch_size = args.global_batch_size
    else:
        # Fallback for cases where these attributes don't exist
        # In FSDP, we might not have these, so we'll use simpler logic
        train_iters = getattr(args, 'lr_decay_iters', None)
        global_batch_size = 1

    if args.lr_decay_iters is None:
        if train_iters is not None:
            args.lr_decay_iters = train_iters
        else:
            # Default to a reasonable value if not specified
            args.lr_decay_iters = 100000

    lr_decay_steps = args.lr_decay_iters * global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * global_batch_size

    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * global_batch_size

    lr_scheduler = FSDPLRScheduler(
        optimizer,
        init_lr=args.lr,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return lr_scheduler
