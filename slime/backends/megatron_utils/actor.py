import socket
import time

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm

if torch.version.hip:
    from vllm.device_allocator.cumem import CuMemAllocator
else:
    from cumem_allocator import CuMemAllocator

from megatron.core import mpu
from sglang.srt.utils import MultiprocessingSerializer
from transformers import AutoConfig, AutoTokenizer

from slime.ray.ppo_actor import TrainRayActor
from slime.utils.distributed_utils import init_process_group
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.timer import Timer, timer

from ..utils.data import process_rollout_data
from .checkpoint import load_checkpoint
from .data import get_data_iterator, log_eval_data, log_perf_data, log_rollout_data
from .initialize import get_gloo_group, init
from .loss import compute_advantages_and_returns
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight_utils import (
    all_gather_param,
    convert_to_hf,
    get_param_info_buckets,
    named_parameters,
    remove_padding,
)


class MegatronTrainRayActor(TrainRayActor):
    def init(self, args, role, with_ref=False):
        super().init(args, role, with_ref)

        wandb_run_id = init(args)
        self.args.wandb_run_id = wandb_run_id

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        if self.args.model_name is None:
            self.model_name = type(self.hf_config).__name__.lower()
        else:
            self.model_name = self.args.model_name
        self.quantization_config = getattr(self.hf_config, "quantization_config", None)
        self.vocab_size = self.tokenizer.vocab_size if self.args.vocab_size is None else self.args.vocab_size

        if self.args.debug_rollout_only:
            Timer().start("train_wait")
            return 0

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args
        )
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
            self.param_info_buckets = get_param_info_buckets(self.args, self.model)

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
        for name, param in named_parameters(self.args, self.model):
            if name not in params_dict:
                params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            params_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def update_gpu_params_dict(self, params_dict):
        for name, param in named_parameters(self.args, self.model):
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

            if self._is_pp_src_rank:
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

        dist.barrier(group=get_gloo_group())

    def set_data_buffer(self, data_buffer):
        self.data_buffer = data_buffer
        if getattr(self.args, "use_wandb", False) and getattr(self.args, "wandb_run_id", None):
            print(f"Updating buffer's wandb run_id to: {self.args.wandb_run_id}")
            ray.get(self.data_buffer.update_wandb_run_id.remote(self.args.wandb_run_id))

    def get_rollout_data(self, rollout_id, rollout_data):
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will recieve the data.
        process_rollout_data(
            rollout_id,
            self.args,
            self.data_buffer,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
            rollout_data=rollout_data,
        )

    def compute_log_prob(
        self,
        model_tag,
        log_probs_data_iterator,
        log_probs_num_microbatches,
        store_prefix="",
        rollout_data=None,
    ):
        # reset data iterator
        for data_iterator in log_probs_data_iterator:
            data_iterator.reset()

        self.update_gpu_params_dict(self.weights[model_tag])

        with timer(f"{store_prefix}log_probs"):
            forward_only(
                self.args,
                self.model,
                log_probs_data_iterator,
                log_probs_num_microbatches,
                store_prefix=store_prefix,
                rollout_data=rollout_data,
            )

    def train(self, rollout_id, with_data_fetching=True):
        Timer().end("train_wait")

        rollout_data = {}

        if self.args.debug_rollout_only:
            # For debug rollout, we just log the data and return.
            if with_data_fetching:
                self.get_rollout_data(rollout_id, rollout_data)
            log_rollout_data(rollout_id, self.args, rollout_data)
            log_perf_data(rollout_id, self.args)
            Timer().start("train_wait")
            return

        if self.args.offload:
            self.wake_up(("model"))

        with timer("train"):
            with timer("data_preprocess"):
                # For async train, we need to separate the data fetching and training.
                if with_data_fetching:
                    self.get_rollout_data(rollout_id, rollout_data)

                # Create data iterator for log_probs and train.
                (
                    log_probs_data_iterator,
                    log_probs_num_microbatches,
                    train_data_iterator,
                    train_num_microbatches,
                ) = get_data_iterator(self.args, self.model, rollout_data)

            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights:
                    self.update_gpu_params_dict(self.weights["ref"])
                    self.compute_log_prob(
                        "ref",
                        log_probs_data_iterator,
                        log_probs_num_microbatches,
                        store_prefix="ref_",
                        rollout_data=rollout_data,
                    )

                self.compute_log_prob(
                    "old_actor" if self.args.keep_old_actor else "actor",
                    log_probs_data_iterator,
                    log_probs_num_microbatches,
                    store_prefix="",
                    rollout_data=rollout_data,
                )
                # when there is old actor, we need to update the model params to actor manually
                if "old_actor" in self.weights:
                    self.update_gpu_params_dict(self.weights["actor"])

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    train_data_iterator,
                    train_num_microbatches,
                )

        log_perf_data(rollout_id, self.args)
        Timer().start("train_wait")

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            return

        # TODO: is logging enough?
        log_eval_data(rollout_id, self.args, self.data_buffer)

    def save_model(self, iteration, with_optimizer=True):
        if self.args.debug_rollout_only:
            return

        if with_optimizer:
            save(iteration, self.model, self.optimizer, self.opt_param_scheduler)
        else:
            save(iteration, self.model, None, None)

    def update_weights_from_distributed(self):
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.reset_prefix_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        for name, param in named_parameters(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name, param, converted_named_tensors, buffer_size, pbar=pbar
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        for name, param in named_parameters(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _update_weight_from_distributed(self, name, param, converted_named_tensors, buffer_size, pbar=None):
        param = all_gather_param(name, param)
        param = remove_padding(name, param, self.vocab_size)
        if not self._is_pp_src_rank:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_distributed(self, name, param, named_tensors, buffer_size, pbar=None):
        param = all_gather_param(name, param)
        param = remove_padding(name, param, self.vocab_size)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(self, named_tensors, pbar=None):
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar=pbar)

    def _update_bucket_weights_from_distributed(self, converted_named_tensors, pbar=None):
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
        pbar.update(1)

    def update_weights_from_tensor(self):
        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.reset_prefix_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
        for param_infos in tqdm(self.param_info_buckets, disable=rank != 0, desc="Update weights"):
            self._update_bucket_weights_from_tensor(param_infos)

    def _update_bucket_weights_from_tensor(self, param_infos):
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        ep_size = mpu.get_expert_model_parallel_world_size()
        rank = dist.get_rank()
        # init params:
        params = []
        for info in param_infos:
            if dist.get_rank() == info.src_rank:
                params.append(
                    torch.nn.Parameter(
                        self.weights["actor"][info.name].to(device=torch.cuda.current_device(), non_blocking=True)
                    )
                )
            else:
                params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
        torch.cuda.synchronize()

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
            param = all_gather_param(info.name, param)
            param = remove_padding(info.name, param, self.vocab_size)
            converted_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
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

    @timer
    def update_weights(self):
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        torch.cuda.empty_cache()
        if not self.args.colocate:
            self.update_weights_from_distributed()
        else:
            self.update_weights_from_tensor()

        dist.barrier(group=get_gloo_group())
        clear_memory()
        print_memory("after update_weights")

        if getattr(self.args, "keep_old_actor", False):
            print("update rollout model on cpu using actor model")
            self.update_cpu_params_dict(self.weights["old_actor"])

    def load_other_checkpoint(self, model_tag, path):
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True
        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        self.weights[model_tag] = {}
        self.update_cpu_params_dict(self.weights[model_tag])
