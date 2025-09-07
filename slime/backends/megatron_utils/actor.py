from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

if torch.version.hip:
    from vllm.device_allocator.cumem import CuMemAllocator
else:
    from torch_memory_saver import torch_memory_saver

from megatron.core import mpu
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.timer import Timer, timer
from slime.utils.wandb_utils import init_wandb_secondary

from ..utils.data import get_data_iterator, process_rollout_data
from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp
from .data import log_perf_data, log_rollout_data
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor, named_parameters


class MegatronTrainRayActor(TrainRayActor):
    def init(self, args, role, wandb_run_id, with_ref=False):
        super().init(args, role, wandb_run_id, with_ref)

        init(args)

        if is_megatron_main_rank():
            init_wandb_secondary(args, wandb_run_id)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

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
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            self.weights["rollout_actor"] = {}
            self.update_cpu_params_dict(self.weights["rollout_actor"])

        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updator = update_weight_cls(
            self.args,
            self.model,
            self.weights,
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            vocab_size=self.tokenizer.vocab_size if self.args.vocab_size is None else self.args.vocab_size,
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload:
            # recover to actor in the end.
            self.update_gpu_params_dict(self.weights["actor"])
            self.sleep(("model"))

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof = None
        if args.use_pytorch_profiler and torch.distributed.get_rank() == 0:
            self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=max(args.profile_step_start - 1, 0),
                    warmup=1 if args.profile_step_start > 0 else 0,
                    active=args.profile_step_end - args.profile_step_start,
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True,
            )
            self.prof.start()

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
        if hasattr(mpu, "destroy_process_groups"):
            mpu.destroy_process_groups()

        if not torch.version.hip:
            torch_memory_saver.pause()
        else:
            allocator = CuMemAllocator.get_instance()
            allocator.sleep(offload_tags=tags)

        print_memory(f"after offload model")

    @timer
    def wake_up(self, tags):
        assert self.args.offload
        print_memory("before wake_up model")

        if isinstance(tags, str):
            tags = (tags,)

        if not torch.version.hip:
            torch_memory_saver.resume()
        else:
            allocator = CuMemAllocator.get_instance()
            allocator.wake_up(tags)

        clear_memory()
        if hasattr(mpu, "reload_process_groups"):
            mpu.reload_process_groups()
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref):
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will recieve the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if "rollout_log_probs" in rollout_data:
            rollout_data["rollout_log_probs"] = [
                torch.tensor(
                    slice_log_prob_with_cp(log_prob, total_length, response_length),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for log_prob, total_length, response_length in zip(
                    rollout_data["rollout_log_probs"], rollout_data["total_lengths"], rollout_data["response_lengths"]
                )
            ]
        return rollout_data

    def compute_log_prob(
        self,
        model_tag,
        data_iterator,
        num_microbatches,
        store_prefix="",
    ):
        self.update_gpu_params_dict(self.weights[model_tag])

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id, rollout_data_ref):
        Timer().end("train_wait")

        if self.args.offload:
            self.wake_up(("model"))

        if self.args.debug_rollout_only:
            # For debug rollout, we just log the data and return.
            rollout_data = self._get_rollout_data(rollout_data_ref)
            log_rollout_data(rollout_id, self.args, rollout_data)
            log_perf_data(rollout_id, self.args)
            Timer().start("train_wait")
            return

        with timer("train"):
            with timer("data_preprocess"):
                rollout_data = self._get_rollout_data(rollout_data_ref)

                # Create data iterator for log_probs and train.
                data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights:
                    rollout_data.update(
                        self.compute_log_prob(
                            "ref",
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                rollout_data.update(
                    self.compute_log_prob(
                        "old_actor" if self.args.keep_old_actor else "actor",
                        data_iterator,
                        num_microbatches,
                        store_prefix="",
                    )
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

            if self.args.use_pytorch_profiler and torch.distributed.get_rank() == 0 and self.prof is not None:
                self.prof.step()

            # Train
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            # Profiling.
            if (
                self.args.use_pytorch_profiler
                and rollout_id == self.args.profile_step_end
                and torch.distributed.get_rank() == 0
                and self.prof is not None
            ):
                self.prof.stop()
                self.prof = None

        # TODO extract to a function during refactor
        if (path_template := self.args.save_debug_train_data) is not None:
            rank = torch.distributed.get_rank()
            path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
            print(f"Save debug train data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    rollout_id=rollout_id,
                    rank=rank,
                    rollout_data=rollout_data,
                ),
                path,
            )

        # update the cpu actor weight to the latest model
        self.update_cpu_params_dict(self.weights["actor"])

        log_perf_data(rollout_id, self.args)
        Timer().start("train_wait")

    def save_model(self, iteration, with_optimizer=True):
        if self.args.debug_rollout_only:
            return

        if with_optimizer:
            save(iteration, self.model, self.optimizer, self.opt_param_scheduler)
        else:
            save(iteration, self.model, None, None)

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        self.weight_updator.connect_rollout_engines(rollout_engines, rollout_engine_lock)
        dist.barrier(group=get_gloo_group())

    @timer
    def update_weights(self):
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.offload and hasattr(mpu, "reload_process_groups"):
            mpu.reload_process_groups()

        with torch_memory_saver.disable() if self.args.offload and not torch.version.hip else nullcontext():
            print_memory("before update_weights")
            self.weight_updator.update_weights()
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                print("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                # First copy rollout_actor to old_actor
                for name in self.weights["old_actor"]:
                    self.weights["old_actor"][name].copy_(self.weights["rollout_actor"][name])
                # Then copy current actor to rollout_actor
                self.update_cpu_params_dict(self.weights["rollout_actor"])

        if self.args.offload and hasattr(mpu, "destroy_process_groups"):
            mpu.destroy_process_groups()

    def load_other_checkpoint(self, model_tag, path):
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights[model_tag] = {}
        self.update_cpu_params_dict(self.weights[model_tag])
