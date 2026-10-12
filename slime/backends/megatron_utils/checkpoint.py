import logging
import os
import re
from contextlib import contextmanager, nullcontext
from copy import copy
from pathlib import Path

import torch

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint as _save_checkpoint_megatron
from megatron.training.global_vars import get_args

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


@contextmanager
def _stage_async_save_on_cpu():
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

    get_save_function_and_args = FileSystemWriterAsync.get_save_function_and_args

    def get_cpu_save_function_and_args(writer):
        save_fn, preload_fn, save_args = get_save_function_and_args(writer)
        if preload_fn is not None:
            # TMS allocations cannot be sent through CUDA IPC. Stage them in the
            # trainer before the persistent worker receives the request; disk
            # writes and distributed finalization still run asynchronously.
            save_args[1] = preload_fn()
        return save_fn, None, save_args

    FileSystemWriterAsync.get_save_function_and_args = get_cpu_save_function_and_args
    try:
        yield
    finally:
        FileSystemWriterAsync.get_save_function_and_args = get_save_function_and_args


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, **kwargs):
    args = get_args()
    stage_on_cpu = args.async_save and args.offload_train and getattr(args, "async_strategy", "mcore") == "mcore"
    with _stage_async_save_on_cpu() if stage_on_cpu else nullcontext():
        _save_checkpoint_megatron(iteration, model, optimizer, opt_param_scheduler, **kwargs)
    if getattr(args, "use_stateless_adam", False) and opt_param_scheduler is not None and dist.get_rank() == 0:
        # Megatron omits the scheduler together with --no-save-optim. Stateless
        # Adam needs no moments, but LR/WD progress must share the model's commit
        # boundary. Store this small payload inside the same checkpoint directory;
        # the joint checkpoint commit waits for and includes it in its manifest.
        path = Path(args.save) / f"iter_{iteration:07d}" / "opt_param_scheduler.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(opt_param_scheduler.state_dict(), temporary)
        temporary.replace(path)


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    native_load = _is_native_model_checkpoint(load_path)
    native_initializer = getattr(args, "prism_pretrained_checkpoint", None)
    # Native SFT components have their own model/manifest layout. They are
    # model-only initializers, even when an explicit iter_* path is supplied.
    # A normal slime --load always keeps precedence over the initializer flag;
    # this also prevents reference/teacher HF loads from loading actor weights.
    if native_load or (load_path is None and native_initializer):
        from slime_plugins.models.prism import load_prism_pretrained

        initializer_args = args
        if native_load:
            initializer_args = copy(args)
            initializer_args.prism_pretrained_checkpoint = load_path
            initializer_args.prism_pretrained_step = getattr(args, "ckpt_step", None)
            if (Path(load_path) / "checkpoint.json").is_file():
                selected_step = (
                    0 if Path(load_path).name == "release" else int(Path(load_path).name.removeprefix("iter_"))
                )
                if initializer_args.prism_pretrained_step not in (None, selected_step):
                    raise ValueError("--ckpt-step disagrees with the explicit native checkpoint directory")
                initializer_args.prism_pretrained_step = None
        result = load_prism_pretrained(ddp_model, initializer_args)
        if optimizer is not None and (args.fp16 or args.bf16):
            optimizer.reload_model_params()
        return result

    if load_path is None:
        raise ValueError("model initialization requires --load or --prism-pretrained-checkpoint")

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        if getattr(args, "use_stateless_adam", False):
            # The master parameters are rebuilt from the saved model. There are
            # no Adam moments to load, and Megatron must skip its optimizer keys.
            args.no_load_optim = True
        result = _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=False,
        )
        if getattr(args, "ckpt_step", None) is not None and not args.finetune and result[0] != args.ckpt_step:
            raise ValueError(
                f"Model loaded step {result[0]}, but checkpoint restoration requires step {args.ckpt_step}. "
                "The Megatron checkpoint loader must honor --ckpt-step (including zero)."
            )
        if getattr(args, "use_stateless_adam", False) and not args.finetune and opt_param_scheduler is not None:
            checkpoint = Path(load_path)
            if not re.fullmatch(r"iter_\d{7,}", checkpoint.name):
                checkpoint /= f"iter_{result[0]:07d}"
            # Loading only the model without this file would silently restart
            # the learning-rate schedule. A missing payload is an invalid resume.
            opt_param_scheduler.load_state_dict(
                torch.load(checkpoint / "opt_param_scheduler.pt", map_location="cpu", weights_only=True)
            )
        return result
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_native_model_checkpoint(path: str | Path | None) -> bool:
    """Identify the split model DCP layout without loading tensors or SFT state."""
    if path is None:
        return False
    candidate = Path(path)
    if (candidate / "checkpoints").is_dir():
        candidate /= "checkpoints"
    if (candidate / "checkpoint.json").is_file() and (candidate / "model").is_dir():
        return True
    tracker = candidate / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        marker = tracker.read_text().strip()
        if marker == "release":
            selected = candidate / "release"
        elif marker.isdecimal():
            selected = candidate / f"iter_{int(marker):07d}"
        else:
            return False
        return (selected / "checkpoint.json").is_file() and (selected / "model").is_dir()
    return False


def _is_megatron_checkpoint(path: str | Path | None) -> bool:
    if path is None or _is_native_model_checkpoint(path):
        return False
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7,}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")
    from slime.backends.megatron_utils.hf_to_megatron import load_hf_weights

    load_hf_weights(args, ddp_model, load_path)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
