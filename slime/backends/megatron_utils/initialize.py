import random

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.training.global_vars import _build_tokenizer, set_args

GLOO_GROUP = None


def _init_gloo_group():
    """Initialize Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo")
    return GLOO_GROUP


def get_gloo_group():
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call _init_gloo_group() first.")
    return GLOO_GROUP


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        seed = seed + (10 * mpu.get_data_parallel_rank(with_context_parallel=False))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)


def _initialize_distributed(args, get_embedding_ranks=None, get_position_embedding_ranks=None):
    """Initialize torch.distributed and core model parallel."""
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
        context_parallel_size=args.context_parallel_size,
        hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
        expert_model_parallel_size=args.expert_model_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        create_gloo_process_groups=args.enable_gloo_process_groups,
    )


def init(args):
    set_args(args)
    # Pytorch distributed.
    _initialize_distributed(args)
    _init_gloo_group()

    # https://github.com/NVIDIA/Megatron-LM/issues/1563
    assert np.__version__.startswith("1."), "Megatron does not support numpy 2.x"

    # Random seeds for reproducibility.
    if args.rank == 0:
        print("> setting random seeds to {} ...".format(args.seed))
    _set_random_seed(
        args.seed,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
    )
    _build_tokenizer(args)
    # We won't use this. initialize to pass some validation in megatron.
    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )

    if args.custom_megatron_init_path:
        from slime.utils.misc import load_function

        custom_init = load_function(args.custom_megatron_init_path)
        custom_init(args)


# TODO shall we use a simpler method to determine which rank to init wandb?
def is_megatron_main_rank():
    return (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    )
