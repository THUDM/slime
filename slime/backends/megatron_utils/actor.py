import ray
import torch
import wandb
import asyncio
import torch.distributed as dist

if torch.version.hip:
    from vllm.device_allocator.cumem import CuMemAllocator
else:
    from cumem_allocator import CuMemAllocator

from megatron.core import mpu

from transformers import AutoConfig, AutoTokenizer
from tracer import vinit, TracePoint, MemTracePoint

from slime.ray.ppo_actor import TrainRayActor
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.timer import Timer, timer
from slime.utils.misc import ActorStatus
from slime.utils.wandb_utils import init_wandb_common

from ..utils.data import process_rollout_data
from .checkpoint import load_checkpoint
from .data import get_data_iterator, log_eval_data, log_perf_data, log_rollout_data
from .initialize import get_gloo_group, init, is_megatron_main_rank
from .loss import compute_advantages_and_returns
from .model import forward_only, initialize_model_and_optimizer, save, train
from megatron.training.global_vars import get_args
from .update_weight_utils import (
    named_parameters,
    UpdateWeightFromTensor,
    UpdateWeightFromDistributed,
)


class MegatronTrainRayActor(TrainRayActor):
    def init(self, args, role, with_ref=False):
        tp = TracePoint(f"task-{args.task_id}: Megatron train actor init", "1")
        tp.begin()
        super().init(args, role, with_ref)
    
        wandb_run_id = init(args)
        env_arg=get_args()
        env_arg.save=args.save
        print("*"*100)
        print(f"{env_arg.save}")
        print("*"*100)
        self.args.wandb_run_id = wandb_run_id

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                c_tp = TracePoint(f"task-{self.args.task_id}: read config and tokenizer", "1")
                c_tp.begin()
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                c_tp.end()
            dist.barrier(group=get_gloo_group(self.args.task_id))

        if self.args.debug_rollout_only:
            Timer().start(self._task_id, "train_wait")
            return 0

        m_op = TracePoint(f"task-{self.args.task_id}: init model and optimizer", "1")
        m_op.begin()
        MemTracePoint.record("before init model and optimizer")
        # torch.cuda.memory._record_memory_history()
        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args
        )
        MemTracePoint.record("after init model and optimizer")
        m_op.end()

        lp = TracePoint(f"task-{self.args.task_id}: update model in cpu", "1")
        lp.begin()
        MemTracePoint.record("before update model in cpu")
        start_rollout_id = loaded_rollout_id + 1
        self.weights = {"actor": {}}
        self.update_cpu_params_dict(self.weights["actor"])
        MemTracePoint.record("after update model in cpu")
        lp.end()

        locp = TracePoint(f"task-{self.args.task_id}: load other model", "1")
        locp.begin()
        if with_ref:
            rp = TracePoint(f"task-{self.args.task_id}: load ref model", "1")
            rp.begin()
            MemTracePoint.record("before load ref model")
            self.load_other_checkpoint("ref", args.ref_load)
            MemTracePoint.record("after load ref model")
            rp.end()

        if self.args.keep_old_actor:
            koap = TracePoint(f"task-{self.args.task_id}: load old actor model", "1")
            koap.begin()
            MemTracePoint.record("before load old actor model")
            self.load_other_checkpoint("old_actor", args.load)
            MemTracePoint.record("after load old actor model")
            koap.end()

        locp.end()

        if self.args.offload:
            offp = TracePoint(f"task-{self.args.task_id}: offload model to cpu", "1")
            offp.begin()
            MemTracePoint.record("before update model in gpu")
            # recover to actor in the end.
            self.update_gpu_params_dict(self.weights["actor"])
            MemTracePoint.record("after update model in gpu")
            MemTracePoint.record("before offload model to cpu")
            self.sleep(("model"))
            MemTracePoint.record("after offload model to cpu")
            offp.end()

        uwp = TracePoint(f"task-{self.args.task_id}: init update weight cls", "1")
        uwp.begin()
        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updator = update_weight_cls(
            self.args,
            self.model,
            self.weights,
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            vocab_size=self.tokenizer.vocab_size if self.args.vocab_size is None else self.args.vocab_size,
        )
        uwp.end()

        MemTracePoint.record("before empty cache")
        # empty cache after initialization
        clear_memory()
        MemTracePoint.record("after empty cache")

        self.rollout_engines = None
        self.data_buffer = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        Timer().start(self._task_id, "train_wait")
        tp.end()
        return start_rollout_id

    def update_cpu_params_dict(self, params_dict):
        with torch.no_grad():
            for name, param in named_parameters(self.args, self.model):
                if name not in params_dict:
                    params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
                params_dict[name].copy_(param.detach(), non_blocking=True)
            torch.cuda.synchronize()

    def update_gpu_params_dict(self, params_dict):
        with torch.no_grad():
            for name, param in named_parameters(self.args, self.model):
                assert name in params_dict
                param.copy_(params_dict[name], non_blocking=True)
            torch.cuda.synchronize()

    def sleep(self, tags):
        with timer(self._task_id, "sleep"):
            tp = TracePoint(f"task-{self.args.task_id}: actor model real sleep", "1")
            tp.begin()
            MemTracePoint.record("before offload actor model")
            assert self.status == ActorStatus.ONLOAD or self.status == ActorStatus.PENDING
            assert self.args.offload
            tags = tags+str(self.args.task_id)
            assert "model"+str(self.args.task_id) in tags
            if isinstance(tags, str):
                tags = (tags,)

            clear_memory()
            print_memory(f"before offload actor model")
            self.update_cpu_params_dict(self.weights["actor"])

            allocator = CuMemAllocator.get_instance()
            allocator.sleep(offload_tags=tags)

            clear_memory()
            print_memory(f"after offload actor model")
            self.status = ActorStatus.OFFLOAD
            MemTracePoint.record("after onload actor model")
            tp.end()

    def wake_up(self, tags):
        with timer(self._task_id, "wake_up"):
            tp = TracePoint(f"task-{self.args.task_id}: actor model real wake up", "1")
            tp.begin()
            MemTracePoint.record("before onload actor model")
            assert self.status == ActorStatus.OFFLOAD
            assert self.args.offload
            tags = tags+str(self.args.task_id) 
            clear_memory()
            print_memory("before wake_up actor model")

            if isinstance(tags, str):
                tags = (tags,)

            allocator = CuMemAllocator.get_instance()
            allocator.wake_up(tags)

            clear_memory()
            print_memory("after wake_up actor model")
            MemTracePoint.record("after offload actor model")
            self.status = ActorStatus.ONLOAD
            tp.end()

    async def set_data_buffer(self, data_buffer):
        tp = TracePoint(f"task-{self.args.task_id}: megatron train actor set data buffer", "1")
        tp.begin()
        self.data_buffer = data_buffer
        if getattr(self.args, "use_wandb", False) and getattr(self.args, "wandb_run_id", None):
            print(f"Updating buffer's wandb run_id to: {self.args.wandb_run_id}")
            await self.data_buffer.update_wandb_run_id.remote(self.args.wandb_run_id)
        tp.end()

    async def get_rollout_data(self, rollout_id, rollout_data):
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will recieve the data.
        await process_rollout_data(
            rollout_id,
            self.args,
            self.data_buffer,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
            rollout_data=rollout_data,
            task_id=self._task_id
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

        with timer(self._task_id, f"{store_prefix}-log_probs"):
            forward_only(
                self.args,
                self.model,
                log_probs_data_iterator,
                log_probs_num_microbatches,
                store_prefix=store_prefix,
                rollout_data=rollout_data,
            )

    async def train(self, rollout_id, with_data_fetching=True):
        if self.args.offload:
            assert self.status == ActorStatus.ONLOAD
        # if self._task_id == 1:
        #     breakpoint()
        Timer().end(self._task_id, "train_wait")
        prepare_train = TracePoint(f"task-{self.args.task_id}: megatron train actor prepare train", "1")
        prepare_train.begin()

        rollout_data = {}
        # breakpoint()        

        if self.args.debug_rollout_only:
            # For debug rollout, we just log the data and return.
            if with_data_fetching:
                data_fetch_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor data fetch", "1")
                data_fetch_trace.begin()
                await self.get_rollout_data(rollout_id, rollout_data)
                data_fetch_trace.end()

            log_rollout_data(rollout_id, self.args, rollout_data)
            log_perf_data(rollout_id, self.args, self._task_id)
            Timer().start(self._task_id, "train_wait")
            return

        with timer(self._task_id, "train"):
            with timer(self._task_id, "data_preprocess"):
                # For async train, we need to separate the data fetching and training.
                if with_data_fetching:
                    data_fetch_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor data fetch", "1")
                    data_fetch_trace.begin()
                    await self.get_rollout_data(rollout_id, rollout_data)
                    data_fetch_trace.end()
                    MemTracePoint.record("megatron train actor after data fetch")

                # Create data iterator for log_probs and train.
                (
                    log_probs_data_iterator,
                    log_probs_num_microbatches,
                    train_data_iterator,
                    train_num_microbatches,
                ) = get_data_iterator(self.args, self.model, rollout_data)

            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights:
                    ref_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor ref log prob", "1")
                    ref_trace.begin()
                    ref_update = TracePoint(f"task-{self.args.task_id}: megatron train actor update ref in gpu", "1")
                    ref_update.begin()
                    MemTracePoint.record("before update ref in gpu")
                    self.update_gpu_params_dict(self.weights["ref"])
                    MemTracePoint.record("after update ref in gpu")
                    ref_update.end()
                    self.compute_log_prob(
                        "ref",
                        log_probs_data_iterator,
                        log_probs_num_microbatches,
                        store_prefix="ref_",
                        rollout_data=rollout_data,
                    )
                    ref_trace.end()

                actor_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor actor log prob", "1")
                actor_trace.begin()
                self.compute_log_prob(
                    "old_actor" if self.args.keep_old_actor else "actor",
                    log_probs_data_iterator,
                    log_probs_num_microbatches,
                    store_prefix="",
                    rollout_data=rollout_data,
                )
                actor_trace.end()
                # when there is old actor, we need to update the model params to actor manually
                if "old_actor" in self.weights:
                    MemTracePoint.record("before update old actor in gpu")
                    self.update_gpu_params_dict(self.weights["actor"])
                    MemTracePoint.record("after update old actor in gpu")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                postprocess_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor data post process", "1")
                postprocess_trace.begin()
                self.rollout_data_postprocess(self.args)
                postprocess_trace.end()

            log_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor log rollout", "1")
            log_trace.begin()
            log_rollout_data(rollout_id, self.args, rollout_data)
            log_trace.end()

            prepare_train.end()
            # Train
            with timer(self._task_id, "actor_train"):
                tp = TracePoint(f"task-{self.args.task_id}: megatron train actor real train", "1")
                tp.begin()
                train(
                    rollout_id,
                    self.args.wandb_run_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    train_data_iterator,
                    train_num_microbatches,
                )
                tp.end()

        log_perf_data(rollout_id, self.args, self._task_id)
        Timer().start(self._task_id, "train_wait")

    async def eval(self, rollout_id):
        if self.args.debug_train_only:
            return

        # TODO: is logging enough?
        await log_eval_data(rollout_id, self.args, self.data_buffer)

    async def save_model(self, iteration, with_optimizer=True):
        tp = TracePoint(f"task-{self.args.task_id}: save model", "1")
        tp.begin()
        MemTracePoint.record("before save model")

        if self.args.debug_rollout_only:
            tp.end()
            return

        if with_optimizer:
            save(iteration, self.model, self.optimizer, self.opt_param_scheduler)
        else:
            save(iteration, self.model, None, None)

        MemTracePoint.record("after save model")
        tp.end()

    async def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        tp = TracePoint(f"task-{self.args.task_id}: connect rollout engines", "1")
        tp.begin()
        
        self.rollout_engines = rollout_engines

        if self.args.debug_train_only or self.args.debug_rollout_only:
            tp.end()
            return

        MemTracePoint.record("before connect weight updator")
        await self.weight_updator.connect_rollout_engines(rollout_engines, rollout_engine_lock)
        MemTracePoint.record("after connect weight updator")

        dist.barrier(group=get_gloo_group(self.args.task_id))
        tp.end()

    async def update_weights(self):
        tp = TracePoint(f"task-{self.args.task_id}: update weights", "1")
        tp.begin()
        MemTracePoint.record("before update weights")

        with timer(self._task_id, "update_weight"):
            if self.args.debug_train_only or self.args.debug_rollout_only:
                return

            MemTracePoint.record("before empty cache")
            torch.cuda.empty_cache()
            MemTracePoint.record("after empty cache")

            MemTracePoint.record("before weight update")
            await self.weight_updator.update_weights()
            MemTracePoint.record("after weight update")

            dist.barrier(group=get_gloo_group(self.args.task_id))
            
            MemTracePoint.record("before clear memory")
            clear_memory()
            MemTracePoint.record("after clear memory")
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                MemTracePoint.record("before update old actor")
                print("update rollout model on cpu using actor model")
                self.update_cpu_params_dict(self.weights["old_actor"])
                MemTracePoint.record("after update old actor")

        tp.end()
    async def update_weights_from_disk(self):
        tp = TracePoint(f"task-{self.args.task_id}: update weights", "1")
        tp.begin()
        MemTracePoint.record("before update weights")

        with timer(self._task_id, "update_weight"):
            if self.args.debug_train_only or self.args.debug_rollout_only:
                return

            MemTracePoint.record("before empty cache")
            torch.cuda.empty_cache()
            MemTracePoint.record("after empty cache")

            MemTracePoint.record("before weight update from disk")
            await self.weight_updator.update_weights_from_disk()
            MemTracePoint.record("after weight update from disk")

            dist.barrier(group=get_gloo_group(self.args.task_id))
            
            MemTracePoint.record("before clear memory")
            clear_memory()
            MemTracePoint.record("after clear memory")
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                MemTracePoint.record("before update old actor")
                print("update rollout model on cpu using actor model")
                self.update_cpu_params_dict(self.weights["old_actor"])
                MemTracePoint.record("after update old actor")

        tp.end()

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
