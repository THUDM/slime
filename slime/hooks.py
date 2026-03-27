"""Lightweight hook system for observability and extensibility.

Provides a context-manager ``hook(op, rollout_id)`` that fires registered
pre/post callbacks around any wrapped operation.  The ``Op`` enum names
every hookable point; callbacks are registered with ``on_pre`` / ``on_post``.

Pre callbacks may return a ``dict`` whose entries are merged into
subsequent pre and post callbacks as extra ``**kwargs``.  The context
manager yields the merged injected dict so that the call-site body can
read values set by pre callbacks.

When no callbacks are registered for an ``Op``, the context manager is
a near-zero-cost no-op.

See: https://github.com/THUDM/slime/issues/1728

Hooked operations in the training loop (train.py / train_async.py)::

    for rollout_id in range(...):
        ITERATION
        ├── EVAL                 # pre-train eval (rollout_id == 0)
        ├── GENERATE             # ray.get(rollout_manager.generate.remote())
        ├── OFFLOAD_ROLLOUT      # ray.get(rollout_manager.offload.remote())
        ├── TRAIN                # ray.get(actor_model.async_train())
        ├── SAVE_MODEL           # actor_model.save_model() + critic_model.save_model()
        ├── OFFLOAD_TRAIN        # actor_model.offload() / clear_memory()
        ├── ONLOAD_ROLLOUT_WEIGHTS
        ├── UPDATE_WEIGHTS       # actor_model.update_weights()
        ├── ONLOAD_ROLLOUT_KV
        ├── EVAL                 # periodic eval
        └── ASYNC_ROLLOUT_SYNC   # train_async.py only: sync before weight update

Example — call-site in train.py::

    from slime.hooks import Op, hook

    with hook(Op.GENERATE, rollout_id):
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Op(Enum):
    """Hookable operations.

    Each member names a point in the training pipeline where pre/post
    callbacks can be registered.  The string value is used as the span name
    in OTel traces and in log messages.
    """

    # head-node operations (train.py / train_async.py)
    ITERATION = "iteration"
    GENERATE = "generate"
    TRAIN = "train"
    UPDATE_WEIGHTS = "update_weights"
    SAVE_MODEL = "save_model"
    EVAL = "eval"
    OFFLOAD_TRAIN = "offload_train"
    OFFLOAD_ROLLOUT = "offload_rollout"
    ONLOAD_ROLLOUT_WEIGHTS = "onload_rollout_weights"
    ONLOAD_ROLLOUT_KV = "onload_rollout_kv"
    ASYNC_ROLLOUT_SYNC = "async_rollout_sync"



_pre: dict[Op, list[Callable[..., Any]]] = {}
_post: dict[Op, list[Callable[..., Any]]] = {}


def on_pre(op: Op, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register a pre-operation callback.

    The callback receives ``rollout_id`` plus any additional kwargs passed
    to ``hook()``, plus state injected by earlier pre callbacks as
    ``**kwargs``.  May return a ``dict`` to inject state into subsequent
    pre and post callbacks.

    Args:
        op: The operation to attach to.
        fn: Callback function.

    Returns:
        ``fn`` unchanged (can be used as a decorator).
    """
    _pre.setdefault(op, []).append(fn)
    logger.debug("Registered pre %s -> %s", op.value, getattr(fn, "__name__", repr(fn)))
    return fn


def on_post(op: Op, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register a post-operation callback.

    The callback receives ``rollout_id`` plus any additional kwargs, plus
    all injected state, plus ``error: BaseException | None`` (``None`` on
    success).  Called after the operation completes, even on exception.

    Args:
        op: The operation to attach to.
        fn: Callback function.

    Returns:
        ``fn`` unchanged (can be used as a decorator).
    """
    _post.setdefault(op, []).append(fn)
    logger.debug("Registered post %s -> %s", op.value, getattr(fn, "__name__", repr(fn)))
    return fn


@contextmanager
def hook(op: Op, rollout_id: int, **kwargs: Any) -> Generator[dict[str, Any], None, None]:
    """Wrap an operation with registered pre/post callbacks.

    Yields a dict of state injected by pre callbacks.  Post callbacks fire
    in the ``finally`` block, so they run even if the body raises.

    When no callbacks are registered for ``op``, this is a near-zero-cost
    no-op that yields an empty dict.

    Args:
        op: The operation being performed.
        rollout_id: Current training iteration index.
        **kwargs: Additional attributes forwarded to callbacks (e.g. rank, gpu).

    Yields:
        Dict of state injected by pre callbacks (empty if none registered).
    """
    pre_fns = _pre.get(op, ())
    post_fns = _post.get(op, ())
    if not pre_fns and not post_fns:
        yield {}
        return

    call_kwargs: dict[str, Any] = {"rollout_id": rollout_id, **kwargs}
    injected: dict[str, Any] = {}
    for fn in list(pre_fns):
        try:
            result = fn(**call_kwargs, **injected)
            if isinstance(result, dict):
                injected.update(result)
        except Exception:
            logger.warning("Pre %s callback %s raised", op.value, getattr(fn, "__name__", repr(fn)), exc_info=True)
    try:
        yield injected
    except BaseException as exc:
        injected["error"] = exc
        raise
    finally:
        injected.setdefault("error", None)
        for fn in list(post_fns):
            try:
                fn(**call_kwargs, **injected)
            except Exception:
                logger.warning(
                    "Post %s callback %s raised", op.value, getattr(fn, "__name__", repr(fn)), exc_info=True
                )


def has_callbacks(op: Op) -> bool:
    """True if any pre or post callbacks are registered for ``op``."""
    return bool(_pre.get(op)) or bool(_post.get(op))


def clear(op: Op | None = None) -> None:
    """Remove callbacks.

    Args:
        op: If given, clear only this operation's callbacks.
            If ``None``, clear all callbacks for all operations.
    """
    if op is None:
        _pre.clear()
        _post.clear()
    else:
        _pre.pop(op, None)
        _post.pop(op, None)
