import os
import socket
import time
from contextlib import nullcontext
from typing import Dict

import ray
import torch
import torch.distributed as dist


if torch.version.hip:
    from vllm.device_allocator.cumem import CuMemAllocator
else:
    from cumem_allocator import CuMemAllocator


from megatron.core import mpu
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.utils import MultiprocessingSerializer
from transformers import AutoConfig, AutoTokenizer

from slime.backends import megatron_utils
from slime.backends.megatron_utils import update_weight_utils
from slime.ray.ray_actor import RayActor
from slime.utils.distributed_utils import init_process_group
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.timer import Timer, timer


@ray.remote(
    num_gpus=1,
    runtime_env={
        "env_vars": {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": "0",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        }
    },
)
class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port):
        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(start_port=20000)

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])

    def init(self, args, role, with_ref=False):
        self.args = args
        self.role = role
        self.with_ref = with_ref

        wandb_run_id = megatron_utils.init(args)
        self.args.wandb_run_id = wandb_run_id

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=megatron_utils.get_gloo_group())

        if self.args.model_name is None:
            self.model_name = type(self.hf_config).__name__.lower()
        else:
            self.model_name = self.args.model_name
        self.quantization_config = getattr(self.hf_config, "quantization_config", None)
        self.vocab_size = self.tokenizer.vocab_size if self.args.vocab_size is None else self.args.vocab_size
        megatron_utils.set_metadata("pad_token_id", self.tokenizer.pad_token_id)

        if self.args.debug_rollout_only:
            Timer().start("train_wait")
            return 0

        allocator = CuMemAllocator.get_instance() if self.args.offload else None

        with allocator.use_memory_pool(tag="model") if allocator else nullcontext():
            (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = (
                megatron_utils.initialize_model_and_optimizer(args, with_optimizer=True)
            )
            clear_memory()
            start_rollout_id = loaded_rollout_id + 1
        self.weights = {"actor": {}}
        self.update_cpu_params_dict(self.weights["actor"])

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if self.args.keep_old_actor:
            self.load_other_checkpoint("old_actor", args.load)

        if self.args.offload:
            # recover to actor in the end.
            self.update_gpu_params_dict(self.weights["actor"])
            self.sleep(("model"))

        if self.args.colocate:
            self.param_info_buckets = update_weight_utils.get_param_info_buckets(self.args, self.model)

        # empty cache after initialization
        clear_memory()

        self.rollout_engines = None
        self.rollout_engine_lock = None
        self.data_buffer = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        Timer().start("train_wait")
        return start_rollout_id

    @torch.no_grad()
    def update_cpu_params_dict(self, params_dict):
        for name, param in update_weight_utils.named_parameters(self.args, self.model):
            if name not in params_dict:
                params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            params_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def update_gpu_params_dict(self, params_dict):
        for name, param in update_weight_utils.named_parameters(self.args, self.model):
            assert name in params_dict
            param.copy_(params_dict[name], non_blocking=True)
        torch.cuda.synchronize()

    @timer
    def sleep(self, tags):
        assert self.args.offload
        assert "model" in tags
        if isinstance(tags, str):
            tags = (tags,)

        clear_memory()
        print_memory(f"before offload model")
        self.update_cpu_params_dict(self.weights["actor"])

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=tags)

        clear_memory()
        print_memory(f"after offload model")

    @timer
    def wake_up(self, tags):
        assert self.args.offload
        clear_memory()
        print_memory("before wake_up model")

        if isinstance(tags, str):
            tags = (tags,)

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        clear_memory()
        print_memory("after wake_up model")

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.colocate:
            # Here we assume the gpu id of rollout engines and train actors are the same.
            for i, engine in enumerate(self.rollout_engines):
                start_rank = i * self.args.rollout_num_gpus_per_engine
                end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
                group_ranks = list(range(start_rank, end_rank))
                new_group = dist.new_group(
                    ranks=group_ranks,
                    backend="gloo",
                )
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_src = start_rank
                    self._ipc_gather_group = new_group
                    self._ipc_engine = engine
        else:
            # For TP:
            #   1. AllGather paramters to rank 0
            #   2. Broadcast parameters from rank 0 to all sglang engines
            self._is_pp_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
            )
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            if self._is_pp_src_rank:
                self._group_name = f"slime-pp_{pp_rank}"

            self._is_ep_src_rank = mpu.get_expert_model_parallel_world_size() > 1 and (
                mpu.get_expert_data_parallel_rank() == 0 and mpu.get_expert_tensor_parallel_rank() == 0
            )
            if self._is_ep_src_rank and not self._is_pp_src_rank:
                self._group_name = f"slime-pp_{pp_rank}_ep_{mpu.get_expert_model_parallel_rank()}"

            if self._is_pp_src_rank or self._is_ep_src_rank:
                master_address = ray._private.services.get_node_ip_address()
                with socket.socket() as sock:
                    sock.bind(("", 0))
                    master_port = sock.getsockname()[1]
                world_size = self.args.rollout_num_gpus + 1

                refs = [
                    engine.init_process_group.remote(
                        master_address,
                        master_port,
                        i * self.args.rollout_num_gpus_per_engine + 1,
                        world_size,
                        self._group_name,
                        backend="nccl",
                    )
                    for i, engine in enumerate(self.rollout_engines)
                ]
                self._model_update_groups = init_process_group(
                    backend="nccl",
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=self._group_name,
                )
                ray.get(refs)

        dist.barrier(group=megatron_utils.get_gloo_group())

    def set_data_buffer(self, data_buffer):
        self.data_buffer = data_buffer
        if getattr(self.args, "use_wandb", False) and getattr(self.args, "wandb_run_id", None):
            print(f"Updating buffer's wandb run_id to: {self.args.wandb_run_id}")
            ray.get(self.data_buffer.update_wandb_run_id.remote(self.args.wandb_run_id))

    def get_rollout_data(self, rollout_id):
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will recieve the data.
        megatron_utils.process_rollout_data(rollout_id, self.args, self.data_buffer)

    def compute_log_prob(
        self,
        model_tag,
        log_probs_data_iterator,
        log_probs_num_microbatches,
        store_prefix="",
    ):
        # reset data iterator
        for data_iterator in log_probs_data_iterator:
            data_iterator.reset()

        self.update_gpu_params_dict(self.weights[model_tag])

        with timer(f"{store_prefix}log_probs"):
            megatron_utils.forward_only(
                self.args,
                self.model,
                log_probs_data_iterator,
                log_probs_num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id, with_data_fetching=True):
        Timer().end("train_wait")

        if self.args.debug_rollout_only:
            # For debug rollout, we just log the data and return.
            if with_data_fetching:
                self.get_rollout_data(rollout_id)
            megatron_utils.log_rollout_data(rollout_id, self.args)
            megatron_utils.log_perf_data(rollout_id, self.args)
            Timer().start("train_wait")
            return

        print_memory("begin train")

        if self.args.offload:
            self.wake_up(("model"))

        with timer("train"):
            with timer("data_preprocess"):
                # For async train, we need to separate the data fetching and training.
                if with_data_fetching:
                    self.get_rollout_data(rollout_id)

                # Create data iterator for log_probs and train.
                (
                    log_probs_data_iterator,
                    log_probs_num_microbatches,
                    train_data_iterator,
                    train_num_microbatches,
                ) = megatron_utils.get_data_iterator(self.args, self.model)

            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights:
                    self.update_gpu_params_dict(self.weights["ref"])
                    self.compute_log_prob(
                        "ref",
                        log_probs_data_iterator,
                        log_probs_num_microbatches,
                        store_prefix="ref_",
                    )

                self.compute_log_prob(
                    "old_actor" if self.args.keep_old_actor else "actor",
                    log_probs_data_iterator,
                    log_probs_num_microbatches,
                )
                # when there is old actor, we need to update the model params to actor manually
                if "old_actor" in self.weights:
                    self.update_gpu_params_dict(self.weights["actor"])

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                megatron_utils.compute_advantages_and_returns(self.args)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            megatron_utils.log_rollout_data(rollout_id, self.args)

            # Train
            with timer("actor_train"):
                megatron_utils.train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    train_data_iterator,
                    train_num_microbatches,
                )

        megatron_utils.log_perf_data(rollout_id, self.args)
        megatron_utils.data.clear_local_storage()
        Timer().start("train_wait")

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            return

        # TODO: is logging enough?
        megatron_utils.log_eval_data(rollout_id, self.args, self.data_buffer)

    def save_model(self, iteration, with_optimizer=True):
        if self.args.debug_rollout_only:
            return

        if with_optimizer:
            megatron_utils.save(iteration, self.model, self.optimizer, self.opt_param_scheduler)
        else:
            megatron_utils.save(iteration, self.model, None, None)

    def update_weights_from_distributed(self):
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.reset_prefix_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=megatron_utils.get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        for name, param in update_weight_utils.named_parameters(self.args, self.model):
            if ".experts." in name:
                continue
            param = update_weight_utils.all_gather_param(name, param)
            param = update_weight_utils.remove_padding(name, param, self.vocab_size)
            if not self._is_pp_src_rank:
                continue
            param_size = param.numel() * param.element_size()
            if buffer_size + param_size > self.args.update_weight_buffer_size:
                self._update_param_from_distributed(converted_named_tensors)
                buffer_size = 0
            converted_named_tensors += update_weight_utils.convert_to_hf(
                self.args, self.model_name, name, param, self.quantization_config
            )
            buffer_size += param_size

        if converted_named_tensors:
            self._update_param_from_distributed(converted_named_tensors)

        dist.barrier(group=megatron_utils.get_gloo_group())

        buffer_size = 0
        for name, param in update_weight_utils.named_parameters(self.args, self.model):
            if ".experts." not in name:
                continue
            param = update_weight_utils.all_gather_param(name, param)
            param = update_weight_utils.remove_padding(name, param, self.vocab_size)
            if not self._is_ep_src_rank:
                continue
            param_size = param.numel() * param.element_size()
            if buffer_size + param_size > self.args.update_weight_buffer_size:
                self._update_param_from_distributed(converted_named_tensors)
                buffer_size = 0
            converted_named_tensors += update_weight_utils.convert_to_hf(
                self.args, self.model_name, name, param, self.quantization_config
            )
            buffer_size += param_size

        if converted_named_tensors:
            self._update_param_from_distributed(converted_named_tensors)

        dist.barrier(group=megatron_utils.get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=megatron_utils.get_gloo_group())

    def _update_param_from_distributed(self, converted_named_tensors):
        # lock the rollout engines to prevent dead lock on broadcast.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in converted_named_tensors],
                dtypes=[param.dtype for _, param in converted_named_tensors],
                shapes=[param.shape for _, param in converted_named_tensors],
                group_name=self._group_name,
            )
            for engine in self.rollout_engines
        ]

        handles = []
        for _, param in converted_named_tensors:
            handles.append(dist.broadcast(param.data, 0, group=self._model_update_groups, async_op=True))
        for handle in handles:
            handle.wait()

        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())

    def update_weights_from_tensor(self):
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        ep_size = mpu.get_expert_model_parallel_world_size()
        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.reset_prefix_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=megatron_utils.get_gloo_group())
        for param_infos in self.param_info_buckets:
            # init params:
            params = []
            for info in param_infos:
                if dist.get_rank() == info.src_rank:
                    params.append(
                        torch.nn.Parameter(self.weights["actor"][info.name].to(device=torch.cuda.current_device()))
                    )
                else:
                    params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))

            # broadcast params across pp ranks
            if pp_size > 1:
                handles = []
                for info, param in zip(param_infos, params):
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                        handles.append(
                            torch.distributed.broadcast(
                                param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                            )
                        )
                for handle in handles:
                    handle.wait()

            # broadcast params across ep ranks
            if ep_size > 1:
                handles = []
                for info, param in zip(param_infos, params):
                    if ".experts." in info.name:
                        src_rank = (
                            info.src_rank
                            if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                            else rank
                        )
                        handles.append(
                            torch.distributed.broadcast(
                                param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True
                            )
                        )
                for handle in handles:
                    handle.wait()

            converted_named_tensors = []
            for info, param in zip(param_infos, params):
                # set tp attrs
                for key, value in info.attrs.items():
                    setattr(param, key, value)
                # gather param
                param = update_weight_utils.all_gather_param(info.name, param)
                param = update_weight_utils.remove_padding(info.name, param, self.vocab_size)
                converted_named_tensors.extend(
                    update_weight_utils.convert_to_hf(
                        self.args, self.model_name, info.name, param, self.quantization_config
                    )
                )
            self._update_converted_params_from_tensor(converted_named_tensors)

    def _update_converted_params_from_tensor(self, converted_named_tensors):
        ipc_handle = MultiprocessingSerializer.serialize(converted_named_tensors, output_str=True)
        ipc_handles = (
            [None] * dist.get_world_size(self._ipc_gather_group) if self._ipc_gather_src == dist.get_rank() else None
        )
        dist.gather_object(
            ipc_handle,
            object_gather_list=ipc_handles,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            ref = self._ipc_engine.update_weights_from_tensor.remote(
                ipc_handles=ipc_handles,
            )
            ray.get(ref)

        converted_named_tensors.clear()
        torch.cuda.empty_cache()

    @timer
    def update_weights(self):
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        torch.cuda.empty_cache()
        if not self.args.colocate:
            self.update_weights_from_distributed()
        else:
            self.update_weights_from_tensor()

        dist.barrier(group=megatron_utils.get_gloo_group())
        clear_memory()
        print_memory("after update_weights")

        if getattr(self.args, "keep_old_actor", False):
            print("update rollout model on cpu using actor model")
            for cpu_module, src_module in zip(self.old_actor, self.model):
                cpu_module.load_state_dict(src_module.state_dict(), strict=True)

    def load_other_checkpoint(self, model_tag, path):
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True
        _, _ = megatron_utils.load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        self.weights[model_tag] = {}
        self.update_cpu_params_dict(self.weights[model_tag])


class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int]],
        num_gpus_per_actor=1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        # Allocate the GPUs for actors w/o instantiating them
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices = pg
        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args, role, with_ref=False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        self.args = args
        return [actor.init.remote(args, role, with_ref=with_ref) for actor in self._actor_handlers]

    def async_init_weight_update_connections(self, rollout):
        """
        Connect rollout engines and actors, e.g. initialize the process group between them
        to update weights after each training stage.
        """
        self.rollout = rollout
        ray.get([actor.set_data_buffer.remote(rollout.data_buffer) for actor in self._actor_handlers])

        return [
            actor.connect_rollout_engines.remote(
                rollout.rollout_engines,
                rollout.rollout_engine_lock,
            )
            for actor in self._actor_handlers
        ]

    def get_rollout_data(self, rollout_id):
        ray.get([actor.get_rollout_data.remote(rollout_id) for actor in self._actor_handlers])

    def async_train(self, rollout_id, with_data_fetching=True):
        """Do one rollout training"""
        return [
            actor.train.remote(rollout_id, with_data_fetching=with_data_fetching) for actor in self._actor_handlers
        ]

    def async_eval(self, rollout_id):
        """Evaluate the model"""
        return [actor.eval.remote(rollout_id) for actor in self._actor_handlers]

    def async_save_model(self, step_id):
        """Save actor model on rank 0."""
        return [actor.save_model.remote(step_id) for actor in self._actor_handlers]

    def async_update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        return [actor.update_weights.remote() for actor in self._actor_handlers]

    def async_offload(self):
        return [actor.sleep.remote(("model")) for actor in self._actor_handlers]
