"""Persist checkpoint boundaries and replay batches across trainer/manager restarts.

ServingCluster owns engines and the live queue. Queue receipts are the authority
for accepted raw/converted data; this journal owns the model/optimizer rollback
boundary and pins its replay window. Training completion advances queue reads,
but only a committed model checkpoint can release those replay pins.
"""

import copy
import hashlib
import itertools
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import ray
import torch

from slime.data.checkpoint import RestorePlan
from slime.data.tensor import materialize_tensor_refs
from slime.data.transport import DiskPayloadRef, RawRolloutRef, pack_rollout_payload, rollout_store
from slime.observability.rollout_data_utils import load_debug_rollout_data

RECOVERY_NAMESPACE = "slime-training-recovery"


def training_recovery_enabled(args):
    # Serving is always retained internally. Trainer replay additionally needs
    # durable batches, regardless of the legacy --use-fault-tolerance flag.
    replay_storage = (
        getattr(args, "rollout_data_transport", "object-store") == "straw"
        or getattr(args, "save_debug_rollout_data", None) is not None
    )
    return (
        replay_storage
        and getattr(args, "train_backend", "megatron") == "megatron"
        and not getattr(args, "rollout_external", False)
        and not any(
            getattr(args, name, False)
            for name in ("debug_train_only", "debug_rollout_only", "load_debug_rollout_data")
        )
    )


def configure_recovery_checkpoint(args):
    """Apply the same recovery policy before and after role-specific overrides."""
    stateless = getattr(args, "use_stateless_adam", False)
    if getattr(args, "no_save_rng", False) or (getattr(args, "no_save_optim", False) and not stateless):
        raise ValueError(
            "Trainer fault tolerance requires optimizer and RNG state; only --use-stateless-adam may omit optimizer state"
        )
    if args.ckpt_format != "torch_dist":
        raise ValueError("Trainer fault tolerance requires --ckpt-format torch_dist for parallelism changes")
    args.ckpt_fully_parallel_save = True
    args.dist_ckpt_optim_fully_reshardable = True


def retained_rollout_configuration(args):
    """Freeze rollout and custom-hook inputs, allowing explicit trainer controls.

    Unknown/custom fields are immutable too: workers keep their original args.
    Adding a new rollout option therefore cannot silently bypass this check.
    """
    mutable = {
        "load",
        "save",
        "ckpt_step",
        "start_rollout_id",
        "finetune",
        "no_load_optim",
        "no_load_rng",
        "save_interval",
        "num_rollout",
        "num_epoch",
        "eval_interval",
        "update_weight_start_version",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "num_layers_per_virtual_pipeline_stage",
        "pipeline_model_parallel_layout",
        "decoder_first_pipeline_num_layers",
        "decoder_last_pipeline_num_layers",
        "micro_batch_size",
        "use_dynamic_batch_size",
        "max_tokens_per_gpu",
        "log_probs_max_tokens_per_gpu",
        "balance_data",
        "balance_by_flops",
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
        "recompute_modules",
        "distribute_saved_activations",
        "sequence_parallel",
        "offload_train",
        "use_distributed_optimizer",
        "optimizer_cpu_offload",
        "optimizer_offload_fraction",
        "use_precision_aware_optimizer",
        "overlap_cpu_optimizer_d2h_h2d",
        "overlap_grad_reduce",
        "overlap_param_gather",
        "ddp_bucket_size",
        "ddp_num_buckets",
        "use_fault_tolerance",
        "distributed_backend",
        "distributed_timeout_minutes",
        "rank",
        "local_rank",
        "world_size",
        "data_parallel_size",
        # Megatron pads the same tokenizer vocabulary to the current TP size.
        "padded_vocab_size",
        "num_layers_per_pipeline_rank",
        "sglang_router_ip",
        "sglang_router_port",
        "sglang_model_routers",
        "distributed_init_method",
        "master_addr",
        "master_port",
        "megatron_config",
        "megatron_config_path",
        "rollout_health_check_interval",
        "rollout_health_check_timeout",
        "rollout_health_check_first_wait",
        "train_env_vars",
        "actor_config",
        "critic_config",
        "ckpt_fully_parallel_save",
        "dist_ckpt_optim_fully_reshardable",
        "distrib_optim_fully_reshardable_mem_efficient",
        "enable_gloo_process_groups",
        "exit_interval",
        "exit_duration_in_mins",
    }
    mutable_prefixes = ("actor_num_", "critic_num_", "wandb_", "tensorboard_", "profiling_", "profile_", "ci_")
    return {
        name: copy.deepcopy(value)
        for name, value in sorted(vars(args).items())
        if name not in mutable and not name.startswith(mutable_prefixes)
    }


def training_session_name(args):
    """Find the same named actors across drivers without depending on trainer layout.

    Prefer an explicit ID or persistent run path. The configuration fallback
    lets runs without replay storage retain their serving cluster as well.
    """
    if identity := getattr(args, "rollout_session_id", None):
        identity = "explicit:" + identity
    elif args.rollout_data_transport == "straw":
        identity = str(Path(args.rollout_data_dir).expanduser().resolve()) + ":" + args.rollout_queue_run_id
    elif path := getattr(args, "save_debug_rollout_data", None) or getattr(args, "save", None):
        identity = str(Path(path).expanduser().resolve())
    else:
        identity = "configuration:" + json.dumps(retained_rollout_configuration(args), sort_keys=True, default=str)
    return "rollout:" + hashlib.sha256(identity.encode()).hexdigest()


@dataclass
class TrainingResume:
    restore_plan: RestorePlan
    configuration: dict
    reused: bool


@dataclass
class ReplayBatch:
    raw: object
    converted: object = None
    batch_id: str | None = None


class TrainingRecovery:
    """Persist the rollback boundary and batches independently of manager lifetime."""

    def __init__(self, args, restore_plan, *, retained_serving=True):
        self.args = args
        self.restore_plan = restore_plan or RestorePlan()
        self.role_configuration = {}
        self.batches = {}
        self.incarnation = uuid.uuid4().hex
        self.loaded = False
        self.checkpoint_step = None
        self.resume_configuration = {
            name: getattr(args, name, None)
            for name in ("load", "save", "ckpt_step", "start_rollout_id", "finetune", "no_load_optim", "no_load_rng")
        }
        self.configuration = retained_rollout_configuration(args)
        self.source_state = None
        # The journal path follows the session identity, not the manager PID,
        # so a replacement manager can find the same checkpoint and batches.
        if args.rollout_data_transport == "straw":
            directory = Path(args.rollout_data_dir) / "training-recovery"
        else:
            directory = Path(args.save_debug_rollout_data.format(rollout_id=0)).parent / ".slime-recovery"
        self.journal = directory / (training_session_name(args).removeprefix("rollout:") + ".pt")
        if self.journal.exists():
            state = torch.load(self.journal, weights_only=False)
            changed = [
                name
                for name in state["configuration"].keys() | self.configuration.keys()
                if name not in state["configuration"]
                or name not in self.configuration
                or state["configuration"][name] != self.configuration[name]
            ]
            if changed:
                raise ValueError("Retained rollout session requires unchanged configuration: " + ", ".join(changed))
            if retained_serving:
                # Reuse the journal only with its live serving/queue owner.
                # Cold startup takes its progress from the checkpoint instead.
                for name, value in state.items():
                    setattr(self, name, value)
            else:
                # A new serving owner must load the checkpoint source/builder
                # handoff instead of treating an old journal as a live queue.
                self._release_batch_storage(state["batches"], state["incarnation"])

    def persist(self):
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        state = {
            name: getattr(self, name)
            for name in (
                "configuration",
                "restore_plan",
                "role_configuration",
                "resume_configuration",
                "batches",
                "incarnation",
                "loaded",
                "checkpoint_step",
                "source_state",
            )
        }
        # Flush the complete new record before atomically replacing the old
        # one; a crash must not expose a half-written recovery journal.
        temporary = self.journal.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.journal)
        # Persist the rename as well as the file contents before acknowledging.
        descriptor = os.open(self.journal.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def reconcile_checkpoint(self):
        """A joint commit may have succeeded just before its manager RPC was lost."""
        if self.args.rollout_data_transport != "straw" or not self.resume_configuration["save"]:
            return
        from slime.data.checkpoint import _read_checkpoint

        root = Path(self.resume_configuration["save"])
        steps = [int(path.stem.removeprefix("committed_")) for path in (root / "rollout").glob("committed_*.json")]
        if steps and (self.checkpoint_step is None or max(steps) > self.checkpoint_step):
            step = max(steps)
            _read_checkpoint(root, step)
            self.checkpoint_committed(step)

    def reconcile_collection(self, controller, branch_id):
        """Recover a collection accepted just before the manager journal write."""
        if not self.loaded:
            # Model loading has not selected the first rollout yet, so this
            # manager cannot have accepted any training data.
            return
        from straw.protocol import Lease, RecordSetRef

        # Generation is sequential: at most the next rollout can have been
        # accepted without reaching remember_raw(). The queue owns that fact;
        # reuse its original receipt before fencing/replaying the old readers.
        rollout_id = max(self.batches, default=self.resume_configuration["start_rollout_id"] - 1) + 1
        task = ray.get(controller.status.remote(f"collection:{branch_id}:{rollout_id}"))
        if task is not None and task["state"] == "completed":
            receipt = ray.get(controller.result.remote(Lease(**task["lease"])))
            self.remember_raw(rollout_id, RawRolloutRef(receipt.result_ref, self.args.rollout_data_dir, receipt))
        # Conversion acceptance is owned by the queue too. A manager can die
        # after that commit and before updating its checkpoint replay window.
        for rollout_id, retained in self.batches.items():
            if retained.converted is not None:
                continue
            batch_id = f"batch:{retained.raw.receipt.commit_id}"
            batch = ray.get(controller.batch.remote(batch_id))
            if batch is not None and batch["ready"]:
                reference = DiskPayloadRef(RecordSetRef.from_dict(batch["ready_ref"]), self.args.rollout_data_dir)
                self.remember_converted(rollout_id, reference, batch_id)

    def resume_role(self, role, configuration):
        if role not in self.role_configuration:
            self.role_configuration[role] = {
                name: getattr(configuration, name, None)
                for name in ("load", "save", "ckpt_step", "finetune", "no_load_optim", "no_load_rng")
            }
            # Actor and critic may use different optimizers. Keep each role's
            # checkpoint load policy even if a later YAML override changes it.
            self.role_configuration[role]["use_stateless_adam"] = getattr(configuration, "use_stateless_adam", False)
        self.persist()
        values = dict(self.role_configuration[role])
        if self.checkpoint_step is not None:
            # Actor/critic YAML may name different paths, but new overrides
            # cannot move a role away from the retained checkpoint boundary.
            values.update(
                load=values["save"],
                ckpt_step=self.checkpoint_step,
                finetune=False,
                no_load_optim=values["use_stateless_adam"],
                no_load_rng=False,
            )
        values["start_rollout_id"] = self.resume_configuration["start_rollout_id"]
        values["update_weight_start_version"] = getattr(self.args, "update_weight_start_version", 0)
        return values

    def remember_raw(self, rollout_id, reference, *, source_state=None):
        # Retain accepted generation before conversion: hooks may fail or the
        # manager may die before it can persist the converted batch.
        if rollout_id in self.batches:
            raise RuntimeError(f"Rollout {rollout_id} is already retained for training recovery")
        self.batches[rollout_id] = ReplayBatch(reference)
        if isinstance(reference, DiskPayloadRef):
            self._retain(rollout_id, reference)
        if source_state is not None:
            self.source_state = source_state
        # Commit the advanced cursor and its accepted batch in one journal
        # write. A cursor-only write could skip this batch after manager death.
        self.persist()

    def remember_converted(self, rollout_id, data, batch_id):
        # Store the global batch, before DP sharding, so a restarted trainer can
        # change parallelism without rerunning reward/conversion hooks.
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
        self.persist()

    def _retain(self, rollout_id, reference):
        # Pin raw and converted data independently of queue consumption. An
        # acknowledged batch may still need replay until its model is saved.
        store, _, lock = rollout_store(self.args)
        raw = self.batches[rollout_id].raw
        roots = [reference.manifest]
        if isinstance(raw, DiskPayloadRef) and raw.manifest not in roots:
            roots.append(raw.manifest)
        with lock:
            store.retain(f"trainer-recovery:{self.incarnation}:{rollout_id}", roots)

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
            self.persist()

    def checkpoint_committed(self, rollout_id):
        # Runtime training completion is insufficient: only a durable training
        # checkpoint lets us release batches needed for replay.
        self.checkpoint_step = rollout_id
        self.resume_configuration.update(
            load=self.resume_configuration["save"],
            ckpt_step=rollout_id,
            start_rollout_id=rollout_id + 1,
            finetune=False,
            no_load_optim=getattr(self.args, "use_stateless_adam", False),
            no_load_rng=False,
        )
        self.release_batches(through=rollout_id)

    def release_batches(self, through=None):
        released = {
            rollout_id: batch for rollout_id, batch in self.batches.items() if through is None or rollout_id <= through
        }
        for rollout_id in released:
            del self.batches[rollout_id]
        # Persist the new boundary before releasing storage. A crash may leave
        # extra retained bytes, but must not leave a journal pointing to GC'd data.
        self.persist()
        self._release_batch_storage(released, self.incarnation)

    def _release_batch_storage(self, batches, incarnation):
        for rollout_id, batch in batches.items():
            if self.args.rollout_data_transport == "straw":
                store, _, lock = rollout_store(self.args)
                with lock:
                    store.release(f"trainer-recovery:{incarnation}:{rollout_id}")
            elif batch.converted:
                Path(batch.converted).unlink(missing_ok=True)
