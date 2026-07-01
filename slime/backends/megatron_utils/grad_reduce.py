"""Per-step grad-reduce sync-func setup for Megatron training.

Split out of ``train()`` so the idempotency guard can be unit-tested without a
full Megatron training step (see ``tests/test_overlap_grad_reduce.py``).
"""

from megatron.core.distributed import DistributedDataParallel as DDP


def configure_overlap_grad_reduce(model, config, args):
    """Set ``no_sync_func`` / ``grad_sync_func`` on ``config`` for
    ``overlap_grad_reduce`` -- exactly once.

    ``config`` is the model config from ``get_model_config(model[0])`` and
    persists across ``train()`` calls. The sync funcs are constant, so they are
    set only on the first call; re-setting them on a later step would trip the
    ``no_sync_func is None`` invariant and crash (#1779). Skipping when they are
    already set is a no-op.

    Args:
        model: Sequence of DDP-wrapped model chunks.
        config: The persistent model ``TransformerConfig``.
        args: Megatron args; reads ``overlap_grad_reduce`` and ``align_grad_reduce``.
    """
    if not (isinstance(model[0], DDP) and args.overlap_grad_reduce):
        return
    if config.no_sync_func is not None:
        return
    config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
    if len(model) == 1:
        config.no_sync_func = config.no_sync_func[0]
    if args.align_grad_reduce:
        config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
        if len(model) == 1:
            config.grad_sync_func = config.grad_sync_func[0]
