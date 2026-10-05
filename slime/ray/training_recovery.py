"""Retain serving and replayable batches across manual trainer restarts."""

import copy
import hashlib
import itertools
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import ray
import torch

from slime.data.checkpoint import RestorePlan
from slime.data.tensor import materialize_tensor_refs
from slime.data.transport import DiskPayloadRef, pack_rollout_payload, rollout_store
from slime.observability.rollout_data_utils import load_debug_rollout_data

logger = logging.getLogger(__name__)
RECOVERY_NAMESPACE = "slime-training-recovery"


def training_recovery_enabled(args):
    return (
        getattr(args, "use_fault_tolerance", False)
        and (
            getattr(args, "rollout_data_transport", "object-store") == "straw"
            or getattr(args, "save_debug_rollout_data", None) is not None
        )
        and not any(
            getattr(args, name, False)
            for name in ("debug_train_only", "debug_rollout_only", "load_debug_rollout_data")
        )
    )


def configure_recovery_checkpoint(args):
    """Apply the same recovery policy before and after role-specific overrides."""
    if getattr(args, "no_save_optim", False) or getattr(args, "no_save_rng", False):
        raise ValueError("Trainer fault tolerance requires checkpoints with optimizer and RNG state")
    if args.ckpt_format != "torch_dist":
        raise ValueError("Trainer fault tolerance requires --ckpt-format torch_dist for parallelism changes")
    args.ckpt_fully_parallel_save = True
    args.dist_ckpt_optim_fully_reshardable = True


def training_session_name(args):
    path = args.rollout_data_dir if args.rollout_data_transport == "straw" else args.save_debug_rollout_data
    identity = str(Path(path).expanduser().resolve())
    if args.rollout_data_transport == "straw":
        identity += ":" + args.rollout_queue_run_id
    return "rollout:" + hashlib.sha256(identity.encode()).hexdigest()


@dataclass
class TrainingResume:
    placements: dict
    restore_plan: RestorePlan
    configuration: dict
    reused: bool


@dataclass
class ReplayBatch:
    raw: object
    converted: object = None
    batch_id: str | None = None


class TrainingRecovery:
    """The live manager owns data retention, driver fencing and resume boundaries.

    The Ray cluster and this manager must remain alive. A model checkpoint is
    the rollback boundary; batches after it stay retained until a successor
    checkpoint commits, including batches trained successfully before a failure.
    """

    def __init__(self, args, restore_plan):
        self.args = args
        self.restore_plan = restore_plan or RestorePlan()
        self.driver_job_id = None
        self.training_actors = {}
        self.role_configuration = {}
        self.batches = {}
        self.incarnation = uuid.uuid4().hex
        self.loaded = False
        self.attachments = 0
        self.checkpoint_step = None
        self.resume_configuration = {
            name: getattr(args, name, None)
            for name in ("load", "save", "ckpt_step", "start_rollout_id", "finetune", "no_load_optim", "no_load_rng")
        }
        self.configuration = self._configuration(args)

    @staticmethod
    def _configuration(args):
        names = {
            "hf_checkpoint",
            "ref_load",
            "prompt_data",
            "data_source_path",
            "rollout_function_path",
            "custom_generate_function_path",
            "custom_rm_path",
            "custom_reward_post_process_path",
            "custom_convert_samples_to_train_data_path",
            "input_key",
            "label_key",
            "metadata_key",
            "tool_key",
            "apply_chat_template",
            "rollout_seed",
            "rollout_shuffle",
            "rollout_batch_size",
            "n_samples_per_prompt",
            "global_batch_size",
            "rollout_data_transport",
            "rollout_data_dir",
            "rollout_queue_run_id",
            "save_debug_rollout_data",
            "rollout_external",
            "rollout_external_engine_addrs",
            "rollout_num_gpus",
            "rollout_num_gpus_per_engine",
            "num_gpus_per_node",
            "colocate",
            "offload_rollout",
            "use_critic",
            "advantage_estimator",
            "rewards_normalization",
            "grpo_std_normalization",
            "reward_key",
            "use_score_centering",
            "use_rollout_routing_replay",
            "sglang_config",
            "sglang_config_path",
        }
        names.update(name for name in vars(args) if name.startswith("sglang_"))
        names.difference_update({"sglang_router_ip", "sglang_router_port", "sglang_model_routers"})
        return {name: copy.deepcopy(getattr(args, name, None)) for name in sorted(names)}

    def validate_attachment(self, args):
        configuration = self._configuration(args)
        changed = [name for name, value in self.configuration.items() if configuration.get(name) != value]
        if changed:
            raise ValueError(
                "Retained rollout session requires unchanged rollout/model configuration: " + ", ".join(changed)
            )
        if self.driver_job_id is not None:
            from ray._private.state import jobs

            previous = next((job for job in jobs() if job["JobID"] == self.driver_job_id), None)
            # Missing GCS state does not prove the previous driver has stopped.
            if previous is None or not previous["IsDead"]:
                raise RuntimeError(f"Rollout session is still owned by training job {self.driver_job_id}")

    def release_trainers(self):
        for actors in self.training_actors.values():
            for actor in actors:
                ray.kill(actor, no_restart=True)
        self.training_actors.clear()

    def register_trainers(self, role, actors, configuration):
        self.training_actors[role] = actors
        if role not in self.role_configuration:
            self.role_configuration[role] = {
                name: getattr(configuration, name, None)
                for name in ("load", "save", "ckpt_step", "finetune", "no_load_optim", "no_load_rng")
            }
        values = dict(self.role_configuration[role])
        if self.checkpoint_step is not None:
            values.update(
                load=values["save"],
                ckpt_step=self.checkpoint_step,
                finetune=False,
                no_load_optim=False,
                no_load_rng=False,
            )
        values["start_rollout_id"] = self.resume_configuration["start_rollout_id"]
        values["update_weight_start_version"] = getattr(self.args, "update_weight_start_version", 0)
        return values

    def remember_raw(self, rollout_id, reference):
        if rollout_id in self.batches:
            raise RuntimeError(f"Rollout {rollout_id} is already retained for training recovery")
        self.batches[rollout_id] = ReplayBatch(reference)
        if isinstance(reference, DiskPayloadRef):
            self._retain(rollout_id, reference)

    def remember_converted(self, rollout_id, data, batch_id):
        if self.args.rollout_data_transport == "straw":
            reference = pack_rollout_payload(data, self.args, rollout_id)
            self._retain(rollout_id, reference)
            store, _, lock = rollout_store(self.args)
            with lock:
                store.release_publications([reference.manifest])
        else:
            path = Path(self.args.save_debug_rollout_data.format(rollout_id=rollout_id) + ".train-recovery.pt")
            temporary = path.with_name(path.name + ".tmp")
            torch.save(materialize_tensor_refs(data), temporary)
            temporary.replace(path)
            reference = str(path)
        batch = self.batches[rollout_id]
        batch.converted, batch.batch_id = reference, batch_id

    def _retain(self, rollout_id, reference):
        store, _, lock = rollout_store(self.args)
        raw = self.batches[rollout_id].raw
        roots = [reference.manifest]
        if isinstance(raw, DiskPayloadRef) and raw.manifest not in roots:
            roots.append(raw.manifest)
        with lock:
            store.retain(f"trainer-recovery:{self.incarnation}:{rollout_id}", roots)

    def retain_replay_shards(self, rollout_id, refs):
        store, _, lock = rollout_store(self.args)
        roots = [ref.manifest for ref in refs]
        with lock:
            store.retain(f"trainer-recovery:{self.incarnation}:{rollout_id}:shards", roots)
            store.release_publications(roots)

    def release_replay_shards(self, rollout_id):
        if self.args.rollout_data_transport == "straw":
            store, _, lock = rollout_store(self.args)
            with lock:
                store.release(f"trainer-recovery:{self.incarnation}:{rollout_id}:shards")

    def load_raw(self, rollout_id):
        reference = self.batches[rollout_id].raw
        if isinstance(reference, DiskPayloadRef):
            from slime.data.transport import load_rollout_samples

            data = load_rollout_samples(reference)
            while data and isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))
            return data
        return load_debug_rollout_data(reference, rollout_id=rollout_id)

    def load_converted(self, rollout_id):
        reference = self.batches[rollout_id].converted
        return reference.load() if isinstance(reference, DiskPayloadRef) else torch.load(reference, weights_only=False)

    def initial_load_completed(self, start_rollout_id):
        if not self.loaded:
            self.resume_configuration["start_rollout_id"] = start_rollout_id
            self.loaded = True

    def checkpoint_committed(self, rollout_id):
        self.checkpoint_step = rollout_id
        self.resume_configuration.update(
            load=self.args.save,
            ckpt_step=rollout_id,
            start_rollout_id=rollout_id + 1,
            finetune=False,
            no_load_optim=False,
            no_load_rng=False,
        )
        self.release_batches(through=rollout_id)

    def release_batches(self, through=None):
        for rollout_id in list(self.batches):
            if through is not None and rollout_id > through:
                continue
            batch = self.batches.pop(rollout_id)
            self.release_replay_shards(rollout_id)
            if self.args.rollout_data_transport == "straw":
                store, _, lock = rollout_store(self.args)
                with lock:
                    store.release(f"trainer-recovery:{self.incarnation}:{rollout_id}")
            elif batch.converted:
                Path(batch.converted).unlink(missing_ok=True)


def create_recoverable_rollout_manager(args, restore_plan):
    from slime.ray.rollout import RolloutManager
    from slime.ray.utils import add_default_ray_env_vars

    manager = RolloutManager.options(
        name=training_session_name(args),
        namespace=RECOVERY_NAMESPACE,
        lifetime="detached",
        get_if_exists=True,
        num_cpus=1,
        num_gpus=0,
        runtime_env={"env_vars": add_default_ray_env_vars()},
        **({"enable_tensor_transport": True} if args.rollout_data_transport == "nixl" else {}),
    ).remote(args, None, restore_plan=restore_plan, persistent=True)
    job_id = ray.get_runtime_context().get_job_id()
    resume = ray.get(manager.attach_training.remote(args, job_id))
    for name, value in resume.configuration.items():
        setattr(args, name, value)
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0
    logger.info("%s rollout session %s", "Reusing" if resume.reused else "Created", training_session_name(args))
    if args.check_weight_update_equal and not resume.reused:
        ray.get(manager.check_weights.remote(action="snapshot"))
        ray.get(manager.check_weights.remote(action="reset_tensors"))
    if args.offload_rollout:
        ray.get(manager.offload.remote())
    return manager, resume.placements, num_rollout_per_epoch, resume.restore_plan
