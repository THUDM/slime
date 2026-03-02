"""
Shared tracking interface for experiment logging backends.

Each backend implements ``init / log / finish``, and :class:`TrackingManager` fans out
calls to every active backend.

To add a new backend:
--------------------
1. Subclass :class:`TrackingBackend`.
2. Register it in :data:`BACKEND_REGISTRY`.
3. Add a corresponding ``--use-<name>`` CLI flag in ``arguments.py``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)



class TrackingBackend(ABC):
    # Interface every logging backend must satisfy.

    @abstractmethod
    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        ...

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        ...

    @abstractmethod
    def finish(self) -> None:
        ...


# Thin adapters for backwards compatibility to keep wandb_utils and tensorboard_utils untouched.
class WandbBackend(TrackingBackend):
    # Delegates to the existing ``wandb_utils`` helpers.

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from .. import wandb_utils

        if primary:
            wandb_utils.init_wandb_primary(args, **kwargs)
        else:
            wandb_utils.init_wandb_secondary(args, **kwargs)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        import wandb

        wandb.log(metrics)

    def finish(self) -> None:
        import wandb

        wandb.finish()


class TensorboardBackend(TrackingBackend):
    # Delegates to the existing ``_TensorboardAdapter`` (part of the TODO).

    _adapter = None

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from ..tensorboard_utils import _TensorboardAdapter

        self._adapter = _TensorboardAdapter(args)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._adapter is not None:
            # Strip step-key entries (e.g. "train/step", "rollout/step") —
            # tensorboard receives step as an explicit argument instead.
            data = {k: v for k, v in metrics.items() if not k.endswith("/step")}
            self._adapter.log(data=data, step=step)

    def finish(self) -> None:
        if self._adapter is not None:
            self._adapter.finish()


class MlflowBackend(TrackingBackend):
    """Delegates to ``mlflow_utils``."""

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from . import mlflow_utils

        mlflow_utils.init_mlflow(args, primary=primary, **kwargs)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        from . import mlflow_utils

        mlflow_utils.log_metrics(metrics, step=step)

    def finish(self) -> None:
        from . import mlflow_utils

        mlflow_utils.finish()


# Registry that maps backend name → (class, args-flag attribute)

BACKEND_REGISTRY: dict[str, tuple[type[TrackingBackend], str]] = {
    "wandb": (WandbBackend, "use_wandb"),
    "tensorboard": (TensorboardBackend, "use_tensorboard"),
    "mlflow": (MlflowBackend, "use_mlflow"),
}


class TrackingManager:
    #Initialises and logs to every enabled backend; used internally by ``logging_utils``.

    def __init__(self) -> None:
        self._backends: list[TrackingBackend] = []

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        for name, (cls, flag) in BACKEND_REGISTRY.items():
            if getattr(args, flag, False):
                logger.info("Initialising tracking backend: %s", name)
                backend = cls()
                backend.init(args, primary=primary, **kwargs)
                self._backends.append(backend)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for backend in self._backends:
            backend.log(metrics, step=step)

    def finish(self) -> None:
        for backend in self._backends:
            try:
                backend.finish()
            except Exception:
                logger.exception(
                    "Error finishing tracking backend %s",
                    type(backend).__name__,
                )
        self._backends.clear()
