import math
from typing import Any

import torch


class StatelessAdam(torch.optim.Optimizer):
    """Adam/AdamW update for the special case where moments are reset every step.

    This optimizer intentionally does not keep ``exp_avg`` or ``exp_avg_sq``.
    Its parameter update matches Adam with zero first and second moments at the
    start of every optimizer step, which is the behavior produced by resetting
    Adam state before each one-step rollout.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        *,
        bias_correction: bool = True,
        adam_w_mode: bool = True,
        maximize: bool = False,
        use_decoupled_grad: bool = False,
        master_weights: bool = False,
        set_grad_none: bool | None = None,
        **_: Any,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if amsgrad:
            raise NotImplementedError("StatelessAdam does not support amsgrad.")
        if master_weights:
            raise NotImplementedError("StatelessAdam relies on Megatron/HybridDeviceOptimizer for master weights.")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            bias_correction=bias_correction,
            adam_w_mode=adam_w_mode,
            maximize=maximize,
            use_decoupled_grad=use_decoupled_grad,
            set_grad_none=True if set_grad_none is None else set_grad_none,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault("step", 0)

    def load_state_dict(self, state_dict) -> None:
        state_dict = dict(state_dict)
        state_dict["state"] = {}
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            adam_w_mode = group.get("adam_w_mode", True)
            bias_correction = group.get("bias_correction", True)
            maximize = group.get("maximize", False)
            use_decoupled_grad = group.get("use_decoupled_grad", False)

            group["step"] = 1

            if bias_correction:
                numerator_scale = 1.0
                denominator_scale = 1.0
            else:
                numerator_scale = 1.0 - beta1
                denominator_scale = math.sqrt(1.0 - beta2)

            for param in group["params"]:
                grad = getattr(param, "decoupled_grad", None) if use_decoupled_grad else param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("StatelessAdam does not support sparse gradients.")

                grad_for_update = grad.neg() if maximize else grad
                if weight_decay != 0 and adam_w_mode:
                    param.mul_(1.0 - lr * weight_decay)
                elif weight_decay != 0:
                    grad_for_update = grad_for_update.add(param, alpha=weight_decay)

                denom = grad_for_update.abs().mul(denominator_scale).add_(eps)
                param.addcdiv_(grad_for_update, denom, value=-lr * numerator_scale)

        return loss
