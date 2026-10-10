"""Convert an accepted rollout once, then give each trainer rank a data view.

Global reward/conversion hooks still see the complete sample batch in their
established order. The accepted conversion stores token/mask tensors once and
shares workers' existing lazy R3/SC records. DP partitions and microbatches are
small read selections over that conversion; a new trainer can rebuild them
without calling hooks again or rewriting tensors.
"""

import json
import logging
from pathlib import Path

import ray
import torch

from slime.data.tensor import TensorRef
from slime.data.transport import DiskPayloadRef, TrainBatchRef, pack_rollout_payload
from slime.observability.rollout_data_utils import (
    tensorize_rollout_data_for_training,
    validate_rollout_routed_experts_for_replay,
)
from slime.utils.data import get_source
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.misc import Box, load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class BatchBuilder:
    def __init__(self, args, *, controller=None):
        self.args = args
        self.controller = controller
        self.rollout_id = -1
        self.raw_ref = None
        self.batch_id = None
        self._plan = None
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )

    def begin(self, samples, *, replay=False):
        """Persist the selection and conversion configuration before invoking hooks."""
        if self.raw_ref is None:
            self.batch_id = None
            self._plan = None
            return None
        from straw.protocol import Lease, RecordSetRef, digest

        controller = self.controller
        self.batch_id = f"batch:{self.raw_ref.receipt.commit_id}"
        configuration = {
            name: getattr(self.args, name, None)
            for name in (
                "custom_reward_post_process_path",
                "custom_convert_samples_to_train_data_path",
                "advantage_estimator",
                "rewards_normalization",
                "grpo_std_normalization",
                "reward_key",
                "global_batch_size",
                "micro_batch_size",
                "use_dynamic_batch_size",
                "balance_data",
                "balance_by_flops",
                "n_samples_per_prompt",
                "rollout_batch_size",
                "use_score_centering",
                "use_rollout_routing_replay",
            )
        }
        identity = {
            "version": 1,
            "raw_digest": self.raw_ref.manifest.digest,
            "sample_indices": [sample.index for sample in samples],
            "rollout_ids": [sample.rollout_id for sample in samples],
            "configuration": configuration,
            "parallel": self.train_parallel_config,
        }
        plan_digest = digest(identity)
        existing = ray.get(controller.batch.remote(self.batch_id))
        if existing:
            self._plan = DiskPayloadRef(RecordSetRef.from_dict(existing["plan_ref"]), self.args.rollout_data_dir)
            plan = self._plan.load()
            # Normal retries must match the original plan. Trainer replay may
            # change parallelism, but still uses the retained plan's identity.
            if not replay and plan["digest"] != plan_digest:
                raise ValueError("An existing batch ID cannot be reused with a different selection or conversion plan")
            if existing["ready"]:
                return self.split_by_dp(
                    DiskPayloadRef(RecordSetRef.from_dict(existing["ready_ref"]), self.args.rollout_data_dir)
                )
        else:
            positions = {self.raw_ref.receipt.position}
            for sample in samples:
                positions.update(getattr(sample, "_queue_source_positions", []))
                if receipt := getattr(sample, "_queue_receipt", None):
                    positions.add(receipt["position"])
                elif lease := getattr(sample, "_queue_lease", None):
                    receipt = ray.get(controller.result.remote(Lease(**lease)))
                    if receipt is not None:
                        positions.add(receipt.position)
            # Samples already carry group receipts/positions; do not reread raw.
            plan = {
                **identity,
                "digest": plan_digest,
                "raw": self.raw_ref,
                "input_positions": sorted(positions),
                "producer_processing": "generation/reward/dynamic and batch filters already applied",
            }
            self._plan = pack_rollout_payload(plan, self.args, self.rollout_id)
        batch = ray.get(controller.plan_batch.remote(self.batch_id, plan["input_positions"], self._plan.manifest))
        self._lease = batch["lease"]
        return None

    def replay_converted(self, data, batch_id):
        """Rebuild only the trainer's schedule from the accepted conversion."""
        self.batch_id = batch_id
        return self.split_by_dp(data)

    def publish_converted(self, data):
        """Write the global conversion once; every DP layout shares its records."""
        data = dict(data)
        data["total_lengths"] = [len(tokens) for tokens in data["tokens"]]
        tensorize_rollout_data_for_training(data)
        # Keep selection/raw dependencies alive with the conversion. This also
        # makes a queue commit sufficient to recover a lost manager reply.
        if self._plan is not None:
            data["_plan"] = self._plan
        metadata = (
            {"task_id": self.batch_id, "attempt_id": self._lease["attempt_id"]} if self.batch_id is not None else None
        )
        reference = pack_rollout_payload(data, self.args, self.rollout_id, metadata=metadata)
        if self.batch_id is not None:
            ray.get(self.controller.ready_batch.remote(self.batch_id, reference.manifest))
        return reference

    def training_completed(self, rollout_id):
        if self.batch_id is not None:
            if rollout_id != self.rollout_id:
                raise ValueError("Training completion does not match the current batch")
            ray.get(self.controller.finish_batch.remote(self.batch_id))

    def save(self, rollout_id):
        if self.args.rollout_data_transport != "straw":
            return
        from straw.reporting import write_report

        state = ray.get(self.controller.training_state.remote())
        if state is None:
            return
        from dataclasses import asdict

        from straw.protocol import RecordSetRef, digest

        from slime.data.transport import rollout_store

        path = Path(self.args.save) / "rollout" / f"builder_state_{rollout_id}.json"
        store, _, lock = rollout_store(self.args)
        refs = [RecordSetRef.from_dict(state[key]) for key in ("state_ref", "progress_ref") if state.get(key)]
        storage_owner = f"checkpoint:{path.resolve()}:{digest([asdict(ref) for ref in refs])}"
        with lock:
            store.retain(storage_owner, refs)
        write_report(
            path,
            {
                "version": 1,
                "run_id": getattr(self.args, "rollout_queue_run_id", None) or "rollout",
                "rollout_id": rollout_id,
                "consumer": state,
                "storage_owner": storage_owner,
            },
        )

    def load(self, rollout_id, *, source_restore=None):
        if self.args.rollout_data_transport != "straw" or not self.args.load:
            return
        if source_restore is not None and source_restore.new_queue:
            # A fork is taken after training completion. Saved pending/ready
            # groups were reintroduced with new receipts; old positions and
            # finished batches belong exclusively to the parent queue.
            self.raw_ref = self.batch_id = self._plan = None
            return
        from straw.protocol import RecordSetRef

        path = Path(self.args.load) / "rollout" / f"builder_state_{rollout_id}.json"
        if not path.exists():
            return
        snapshot = json.loads(path.read_text())
        if snapshot["version"] != 1 or snapshot["run_id"] != (
            getattr(self.args, "rollout_queue_run_id", None) or "rollout"
        ):
            raise ValueError("BatchBuilder checkpoint run/schema differs")
        state = snapshot["consumer"]
        ray.get(
            self.controller.restore_training_state.remote(
                RecordSetRef.from_dict(state["state_ref"]),
                state["fetch_cursor"],
                state["processed_cursor"],
                source_restore.source_ref if source_restore is not None else None,
            )
        )

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            if not samples:
                return raw_rewards, []

            group_indices = [sample.group_index for sample in samples]
            if all(group is None for group in group_indices):
                # Legacy generators may omit group_index for an ordered, fixed-size
                # batch. Split episodes need explicit query IDs: row offsets no
                # longer identify prompt boundaries after fan-out.
                group_size = self.args.n_samples_per_prompt
                rollout_ids = [sample.rollout_id for sample in samples if sample.rollout_id is not None]
                if group_size <= 0 or len(samples) % group_size != 0 or len(set(rollout_ids)) != len(rollout_ids):
                    raise ValueError("Reward normalization requires group_index for uneven groups or split rollouts")
                group_indices = [i // group_size for i in range(len(samples))]
            elif any(group is None for group in group_indices):
                raise ValueError("Reward normalization requires group_index on every sample or on none of them")

            groups = {}
            for i, (sample, group) in enumerate(zip(samples, group_indices, strict=True)):
                # Missing rollout IDs denote independent rows, as in convert().
                episode = ("rollout", sample.rollout_id) if sample.rollout_id is not None else ("row", i)
                groups.setdefault(group, {}).setdefault(episode, []).append(i)

            rewards = torch.empty(len(samples), dtype=torch.float)
            for group, episodes in groups.items():
                episode_rows = list(episodes.values())
                for rows in episode_rows:
                    if any(raw_rewards[i] != raw_rewards[rows[0]] for i in rows[1:]):
                        raise ValueError(
                            f"Inconsistent rewards for group_index={group}, "
                            f"rollout_id={samples[rows[0]].rollout_id}; split rows must share one episode reward. "
                            "Use custom_reward_post_process_path for segment-specific rewards."
                        )

                # Each sampled episode contributes once, regardless of how many
                # training rows context rewrites produced. Broadcast back without
                # reordering the rows consumed by the trainer.
                group_rewards = torch.tensor([raw_rewards[rows[0]] for rows in episode_rows], dtype=torch.float)
                group_rewards = group_rewards - group_rewards.mean()
                if (
                    self.args.advantage_estimator in ["grpo", "gspo", "cispo"]
                    and self.args.grpo_std_normalization
                    and len(episode_rows) > 1
                ):
                    group_rewards = group_rewards / (group_rewards.std() + 1e-6)
                for rows, reward in zip(episode_rows, group_rewards, strict=True):
                    rewards[rows] = reward

            return raw_rewards, rewards.tolist()

        return raw_rewards, raw_rewards

    def convert(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        rollout_ids = [sample.rollout_id for sample in samples]
        existed_rollout_id_values = set(rid for rid in rollout_ids if rid is not None)
        tmp_id = 0
        for i in range(len(rollout_ids)):
            if rollout_ids[i] is None:
                while tmp_id in existed_rollout_id_values:
                    tmp_id += 1
                rollout_ids[i] = tmp_id
                existed_rollout_id_values.add(tmp_id)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "rollout_ids": rollout_ids,
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # Per-rollout aggregate, precomputed at the step level (where we can
        # see every sample of every rollout) and broadcast per-sample so the
        # per-mb loss reducer uses the correct whole-rollout denominator even
        # when a rollout's samples land in different micro-batches (first-fit
        # packing can split a rollout across mbs):
        #
        #   ``rollout_mask_sums[i]`` — sum of loss-mask totals over every
        #   sample in sample i's rollout. Used as the reducer's denominator
        #   so summing partial contributions across mbs yields one
        #   token-weighted mean per rollout.
        rollout_id_list = train_data["rollout_ids"]
        mask_sums_per_sample = [sum(m) for m in loss_masks]
        rollout_total_mask: dict[int, int] = {}
        for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
            rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
        train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]

        # Overwrite raw_reward when available. Mixed-source batches may only
        # populate this field for a subset of samples (e.g. SWE but not code).
        if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
            train_data["raw_reward"] = [
                sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
                for sample in samples
            ]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if getattr(self.args, "use_score_centering", False):
            from slime.utils.score_centering import validate_sampler_top_p, validate_sampler_topk

            for sample in samples:
                if sample.rollout_log_probs is None or len(sample.rollout_log_probs) != sample.response_length:
                    raise ValueError("Score centering requires sampler logprobs for every response token.")
                if self.args.rollout_top_p < 1:
                    if sample.response_length == 0 and sample.rollout_top_p_token_ids is None:
                        sample.rollout_top_p_token_ids = torch.empty(0, dtype=torch.int32)
                        sample.rollout_top_p_token_offsets = torch.zeros(1, dtype=torch.int32)
                        sample.rollout_top_p_log_probs = torch.empty(0, dtype=torch.float32)
                    validate_sampler_top_p(
                        sample.rollout_top_p_token_ids,
                        sample.rollout_top_p_token_offsets,
                        sample.rollout_top_p_log_probs,
                        sample.response_length,
                        sample.loss_mask,
                        sample.tokens[-sample.response_length :] if sample.response_length else [],
                        sample.rollout_log_probs,
                    )
                else:
                    validate_sampler_topk(sample, self.args.score_centering_top_k)
            if self.args.rollout_top_p < 1:
                train_data["rollout_top_p_log_probs"] = [sample.rollout_top_p_log_probs for sample in samples]
            else:
                train_data["rollout_topk_token_ids"] = [sample.rollout_topk_token_ids for sample in samples]
                train_data["rollout_topk_log_probs"] = [sample.rollout_topk_log_probs for sample in samples]

        if getattr(self.args, "rollout_top_p", 1.0) != 1.0:
            for sample in samples:
                assert sample.rollout_top_p_token_ids is not None
                assert sample.rollout_top_p_token_offsets is not None
                assert len(sample.rollout_top_p_token_offsets) == sample.response_length + 1, (
                    f"top-p token offsets length {len(sample.rollout_top_p_token_offsets)} "
                    f"!= response length + 1 {sample.response_length + 1}"
                )
                offset_end = int(sample.rollout_top_p_token_offsets[-1:][0])
                assert offset_end == len(
                    sample.rollout_top_p_token_ids
                ), f"top-p token offsets[-1] {offset_end} != token ids length {len(sample.rollout_top_p_token_ids)}"
            train_data["rollout_top_p_token_ids"] = [sample.rollout_top_p_token_ids for sample in samples]
            train_data["rollout_top_p_token_offsets"] = [sample.rollout_top_p_token_offsets for sample in samples]

        routed_experts_present = [sample.rollout_routed_experts is not None for sample in samples]
        dead_sample_indices = {
            idx
            for idx, (present, sample) in enumerate(zip(routed_experts_present, samples, strict=True))
            if not present and sample.loss_mask is not None and not any(sample.loss_mask)
        }
        live_missing = [
            idx for idx, present in enumerate(routed_experts_present) if not present and idx not in dead_sample_indices
        ]
        routing_replay_enabled = getattr(self.args, "use_rollout_routing_replay", False)
        if live_missing and (routing_replay_enabled or any(routed_experts_present)):
            raise ValueError(f"Rollout routed experts are missing for live sample indices {live_missing[:32]}.")

        if any(routed_experts_present) or (routing_replay_enabled and dead_sample_indices):
            dtype = torch.uint8 if self.args.num_experts <= 256 else torch.int32
            routed_experts = [
                (
                    (
                        sample.rollout_routed_experts
                        if isinstance(sample.rollout_routed_experts, TensorRef)
                        else sample.materialize_rollout_routed_experts()
                    )
                    if present
                    else torch.zeros(
                        (max(0, len(sample.tokens) - 1), self.args.num_layers, self.args.moe_router_topk),
                        dtype=dtype,
                    )
                )
                for sample, present in zip(samples, routed_experts_present, strict=True)
            ]
            if routing_replay_enabled and any(routed_experts_present):
                captured_pairs = [
                    (experts, max(0, len(sample.tokens) - 1))
                    for sample, experts, present in zip(samples, routed_experts, routed_experts_present, strict=True)
                    if present
                ]
                validate_rollout_routed_experts_for_replay(
                    [experts for experts, _ in captured_pairs],
                    self.args,
                    expected_rows=[rows for _, rows in captured_pairs],
                )
            if dead_sample_indices:
                logger.warning(
                    "Using all-zero routed experts for %d fully loss-masked samples (indices %s).",
                    len(dead_sample_indices),
                    sorted(dead_sample_indices)[:16],
                )
            train_data["rollout_routed_experts"] = routed_experts

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        if samples[0].metadata is not None:
            train_data["source_names"] = [get_source(sample) for sample in samples]

        return train_data

    def split_by_dp(self, data):
        """Build a schedule over one immutable conversion, without per-rank writes."""
        transport = self.args.rollout_data_transport
        if transport == "straw":
            # Scheduling needs lengths and rollout identity only. In particular,
            # replay must not read the full token or R3 payload on the manager.
            reference = data if isinstance(data, DiskPayloadRef) else self.publish_converted(data)
            data = reference.load(selection={"total_lengths": None, "rollout_ids": None})
        total_lengths = data.get("total_lengths")
        if total_lengths is None:
            total_lengths = [len(tokens) for tokens in data["tokens"]]
        data["total_lengths"] = total_lengths
        partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
            self.args,
            self.train_parallel_config,
            total_lengths,
            global_batch_size=self.args.global_batch_size,
            rollout_indices=data["rollout_ids"],
        )
        fields = [
            "tokens",
            "multimodal_train_inputs",
            "response_lengths",
            "rewards",
            "truncated",
            "loss_masks",
            "round_number",
            "sample_indices",
            "rollout_ids",
            "rollout_mask_sums",
            "rollout_log_probs",
            "rollout_topk_token_ids",
            "rollout_topk_log_probs",
            "rollout_top_p_token_ids",
            "rollout_top_p_token_offsets",
            "rollout_top_p_log_probs",
            "rollout_routed_experts",
            "source_names",
            "prompt",
            "teacher_log_probs",
        ]
        if transport == "straw":
            from straw.protocol import digest

            plan_digest = digest(
                {
                    "conversion": reference.manifest.digest,
                    "partitions": partitions,
                    "micro_batch_indices": micro_batch_indices,
                    "global_batch_sizes": global_batch_sizes,
                }
            )
        result = []
        for rank, partition in enumerate(partitions):
            schedule = {
                "partition": partition,
                "global_batch_sizes": global_batch_sizes,
                "num_microbatches": num_microbatches,
                "micro_batch_indices": micro_batch_indices[rank],
            }
            if transport == "straw":
                selection = {key: partition for key in fields}
                # Training's pass-rate and load metrics use the global arrays;
                # all per-sample tensors are selected before storage reads.
                selection.update(raw_reward=None, total_lengths=None)
                value = TrainBatchRef(
                    reference.manifest,
                    reference.root,
                    self.batch_id,
                    rank,
                    plan_digest,
                    selection=selection,
                    schedule=schedule,
                )
            else:
                value = {key: [data[key][index] for index in partition] for key in fields if key in data}
                value.update({key: data[key] for key in ("raw_reward", "total_lengths") if key in data})
                value.update(schedule)
                tensorize_rollout_data_for_training(value)
            if transport == "nixl":
                result.append(Box(ray.put(value, _tensor_transport="nixl")))
            elif transport in {"straw", "object-store"}:
                result.append(Box(ray.put(value)))
            else:
                raise ValueError(f"Unsupported rollout data transport: {transport!r}")
        return result
