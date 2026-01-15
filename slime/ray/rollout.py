import logging
import multiprocessing
import random
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import ray
import torch
import wandb
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.ray.rollout_data_source_factory import create_rollout_data_source
from slime.rollout.base_types import call_rollout_fn
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import find_available_port, get_host_info, init_http_client
from slime.utils.iter_utils import group_by
from slime.utils.metric_checker import MetricChecker
from slime.utils.metric_utils import compute_pass_rate, compute_statistics, dict_add_prefix
from slime.utils.misc import load_function
from slime.utils.ray_utils import Box
from slime.utils.types import Sample
from slime.utils.wandb_utils import init_wandb_secondary

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg, wandb_run_id):
        self.args = args
        self.pg = pg
        _start_router(args)
        # TODO make args immutable
        init_wandb_secondary(
            args, wandb_run_id, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        )
        init_http_client(args)

        # === Create data source using factory (supports in_process/http/none) ===
        self.data_source = create_rollout_data_source(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        # === Off-Policy tracking ===
        self.current_policy_version = 0

        # === Staleness Control ===
        if args.max_staleness >= 0:
            from slime.utils.offpolicy_utils import StalenessController
            self.staleness_controller = StalenessController(
                max_staleness=args.max_staleness,
                batch_size=args.rollout_batch_size * args.n_samples_per_prompt,
            )
            print(f"[Staleness Control] Initialized with max_staleness={args.max_staleness}, batch_size={args.rollout_batch_size * args.n_samples_per_prompt}")
        else:
            self.staleness_controller = None
            print("[Staleness Control] Disabled (max_staleness < 0)")

        if self.args.debug_train_only:
            self.all_rollout_engines = []
        else:
            num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
            num_engines = args.rollout_num_gpus // num_gpu_per_engine
            self.all_rollout_engines = [None] * num_engines
        self.num_new_engines = init_rollout_engines(args, pg, self.all_rollout_engines)
        self.nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

        self._metric_checker = MetricChecker.maybe_create(args)
        if self.args.use_fault_tolerance:
            self._health_monitor = RolloutHealthMonitor(self, args)

    def dispose(self):
        if self._metric_checker is not None:
            self._metric_checker.dispose()

    # TODO maybe rename "rollout_engines" and "all_rollout_engines" later
    @property
    def rollout_engines(self):
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        return self.all_rollout_engines[:: self.nodes_per_engine]

    def get_rollout_engines_and_lock(self):
        return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        # === Check staleness budget before generation ===
        if self.staleness_controller is not None:
            if not self.staleness_controller.can_submit_request():
                print(
                    f"[Staleness Control] WARNING: Rollout {rollout_id} would exceed staleness budget "
                    f"(num_generated={self.staleness_controller.num_generated}, "
                    f"policy_version={self.staleness_controller.current_policy_version}, "
                    f"max_staleness={self.staleness_controller.max_staleness}). "
                    f"Proceeding anyway, but this may degrade off-policy performance."
                )

        monitor_started = self.args.use_fault_tolerance and self._health_monitor.start()
        start_time = time.time()
        try:
            # === Step 1: Generate new rollout data ===
            data, metrics = self._get_rollout_data(rollout_id=rollout_id)
            self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
            _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)

            # === Step 2: Add to buffer (automatic based on buffer_enabled) ===
            # data is list[Sample] (flattened), add_samples will auto-group it
            self.data_source.add_samples(data)

            # === Step 3: Log buffer stats ===
            buffer_stats = self.data_source.get_buffer_stats()
            if buffer_stats["enabled"]:
                # print(f"[Buffer] Added {len(data)} samples. Stats: {buffer_stats}") # 原有的简单打印

                # ================= 修复开始 =================
                # 1. 无论 wandb 状态如何，先构建完整的 metrics 字典
                wandb_metrics = {
                    "buffer/size": buffer_stats["buffer_size"],
                    "buffer/utilization": buffer_stats["buffer_utilization"],
                    "rollout_id": rollout_id,
                }

                # === NEW: Off-policy metrics ===
                # Policy version tracking
                if "current_policy_version" in buffer_stats:
                    wandb_metrics["buffer/current_policy_version"] = buffer_stats["current_policy_version"]

                if "min_policy_version" in buffer_stats:
                    wandb_metrics["buffer/min_policy_version"] = buffer_stats["min_policy_version"]
                    wandb_metrics["buffer/max_policy_version"] = buffer_stats["max_policy_version"]
                    wandb_metrics["buffer/avg_policy_version"] = buffer_stats["avg_policy_version"]

                # Staleness metrics
                if "min_staleness" in buffer_stats:
                    wandb_metrics["buffer/min_staleness"] = buffer_stats["min_staleness"]
                    wandb_metrics["buffer/max_staleness"] = buffer_stats["max_staleness"]
                    wandb_metrics["buffer/avg_staleness"] = buffer_stats["avg_staleness"]

                # Sample reuse metrics
                if "avg_reuse_count" in buffer_stats:
                    wandb_metrics["buffer/avg_reuse_count"] = buffer_stats["avg_reuse_count"]
                    wandb_metrics["buffer/max_reuse_count_observed"] = buffer_stats["max_reuse_count_observed"]

                # Sampling strategy metrics
                if "strategy_strategy" in buffer_stats:
                    wandb_metrics["buffer/strategy"] = buffer_stats["strategy_strategy"]

                # === NEW: Priority sampling specific metrics ===
                # Priority configuration
                if "strategy_priority_metric" in buffer_stats:
                    wandb_metrics["buffer/priority_metric"] = buffer_stats["strategy_priority_metric"]
                    wandb_metrics["buffer/priority_weight"] = buffer_stats["strategy_priority_weight"]
                    wandb_metrics["buffer/staleness_penalty"] = buffer_stats["strategy_staleness_penalty"]

                # Base score statistics (raw)
                if "strategy_base_score_mean" in buffer_stats:
                    wandb_metrics["buffer/base_score_mean"] = buffer_stats["strategy_base_score_mean"]
                    wandb_metrics["buffer/base_score_min"] = buffer_stats["strategy_base_score_min"]
                    wandb_metrics["buffer/base_score_max"] = buffer_stats["strategy_base_score_max"]

                # Staleness statistics (raw)
                if "strategy_staleness_mean" in buffer_stats:
                    wandb_metrics["buffer/staleness_stat_mean"] = buffer_stats["strategy_staleness_mean"]
                    wandb_metrics["buffer/staleness_stat_min"] = buffer_stats["strategy_staleness_min"]
                    wandb_metrics["buffer/staleness_stat_max"] = buffer_stats["strategy_staleness_max"]

                # Latest sampling round statistics
                if "strategy_latest_base_score_raw_mean" in buffer_stats:
                    wandb_metrics["buffer/latest_base_score_raw_mean"] = buffer_stats["strategy_latest_base_score_raw_mean"]
                    wandb_metrics["buffer/latest_base_score_raw_std"] = buffer_stats["strategy_latest_base_score_raw_std"]
                    wandb_metrics["buffer/latest_base_score_normalized_mean"] = buffer_stats["strategy_latest_base_score_normalized_mean"]

                if "strategy_latest_staleness_raw_mean" in buffer_stats:
                    wandb_metrics["buffer/latest_staleness_raw_mean"] = buffer_stats["strategy_latest_staleness_raw_mean"]
                    wandb_metrics["buffer/latest_staleness_raw_std"] = buffer_stats["strategy_latest_staleness_raw_std"]
                    wandb_metrics["buffer/latest_staleness_normalized_mean"] = buffer_stats["strategy_latest_staleness_normalized_mean"]

                if "strategy_latest_final_score_mean" in buffer_stats:
                    wandb_metrics["buffer/latest_final_score_mean"] = buffer_stats["strategy_latest_final_score_mean"]
                    wandb_metrics["buffer/latest_final_score_std"] = buffer_stats["strategy_latest_final_score_std"]
                    wandb_metrics["buffer/latest_final_score_min"] = buffer_stats["strategy_latest_final_score_min"]
                    wandb_metrics["buffer/latest_final_score_max"] = buffer_stats["strategy_latest_final_score_max"]

                # Normalization configuration
                if "strategy_normalize_scores" in buffer_stats:
                    wandb_metrics["buffer/normalize_scores"] = 1 if buffer_stats["strategy_normalize_scores"] else 0
                    if buffer_stats.get("strategy_normalization_method"):
                        wandb_metrics["buffer/normalization_method"] = buffer_stats["strategy_normalization_method"]

                # 2. 强制打印到控制台 (Stdout Log)
                # 这样即使在 Offline 模式或未启用 WandB 时，你也能在日志文件中看到这些数据
                print(f"[Buffer] Added {len(data)} samples.")
                print(f"[Buffer Metrics] {wandb_metrics}")

                # 3. 安全地记录到 WandB
                # 只要 wandb 被初始化（无论是 online 还是 offline 模式），run 都不为 None
                if wandb.run is not None:
                    wandb.log(wandb_metrics)
                # ================= 修复结束 =================

            # === Step 4: Sample data for training (mixes buffer + new data automatically) ===
            # 🔧 IMPORTANT: Use get_training_samples() for training to avoid getting incomplete prompts
            # get_training_samples() ONLY returns complete samples from buffer, no fallback to dataset
            # get_samples() is used for rollout generation and may return PENDING prompts

            if hasattr(self.data_source, 'get_training_samples') and self.data_source.buffer_enabled:
                # === FIX 4: Calculate correct sample count for off-policy mode ===
                # In off-policy mode, we need to sample enough groups to satisfy global_batch_size
                # num_samples is in GROUPS (not individual samples)
                # Each group has n_samples_per_prompt samples
                import math
                num_groups_needed = math.ceil(self.args.global_batch_size / self.args.n_samples_per_prompt)

                print(f"[Off-Policy Training] Sampling {num_groups_needed} groups from buffer "
                      f"(global_batch_size={self.args.global_batch_size}, "
                      f"n_samples_per_prompt={self.args.n_samples_per_prompt}, "
                      f"total_samples={num_groups_needed * self.args.n_samples_per_prompt})")

                # Use dedicated method for training that only samples from buffer
                train_data_samples = self.data_source.get_training_samples(
                    num_samples=num_groups_needed
                )

                # If buffer is empty/exhausted, skip this training step
                if len(train_data_samples) == 0:
                    print(f"[Training] Buffer exhausted, no samples available for training. Skipping this step.")
                    print(f"[Training] Next rollout will generate new data to refill buffer.")
                    return None  # Signal to skip training
            else:
                # Fallback to regular get_samples for backward compatibility
                # In on-policy mode, use rollout_batch_size (original behavior)
                train_data_samples = self.data_source.get_samples(
                    num_samples=self.args.rollout_batch_size
                )

            # 🔧 DEFENSIVE CHECK: Verify buffer samples are complete
            if self.data_source.buffer_enabled and len(train_data_samples) > 0:
                incomplete_groups = []
                for i, group in enumerate(train_data_samples):
                    for j, sample in enumerate(group):
                        if sample.status == Sample.Status.PENDING and (not hasattr(sample, 'response') or not sample.response):
                            incomplete_groups.append((i, j, sample))

                if len(incomplete_groups) > 0:
                    print(f"[CRITICAL WARNING] Found {len(incomplete_groups)} incomplete samples from buffer!")
                    for i, j, sample in incomplete_groups[:5]:  # Show first 5
                        print(f"  Group {i}, Sample {j}: "
                              f"group_index={getattr(sample, 'group_index', '?')}, "
                              f"index={getattr(sample, 'index', '?')}, "
                              f"status={sample.status}, "
                              f"has_response={hasattr(sample, 'response') and bool(sample.response)}")

                    raise RuntimeError(
                        f"Buffer integrity error: {len(incomplete_groups)} incomplete samples detected! "
                        f"This indicates buffer contamination. Check buffer filtering logic."
                    )

            # === Step 5: Flatten grouped samples for _convert_samples_to_train_data ===
            # _convert_samples_to_train_data expects list[Sample], not list[list[Sample]]
            if len(train_data_samples) > 0 and isinstance(train_data_samples[0], list):
                # Flatten: list[list[Sample]] -> list[Sample]
                flattened_samples = []
                for group in train_data_samples:
                    flattened_samples.extend(group)
                train_data_samples = flattened_samples

            # === FIX 5: Trim training data to exact global_batch_size ===
            # In off-policy mode, we might sample slightly more than global_batch_size
            # (due to rounding up to complete groups). Trim to exact size to ensure
            # get_data_iterator assertions pass.
            if self.data_source.buffer_enabled and len(train_data_samples) > self.args.global_batch_size:
                original_len = len(train_data_samples)
                train_data_samples = train_data_samples[:self.args.global_batch_size]
                print(f"[Off-Policy Training] Trimmed training data from {original_len} to {len(train_data_samples)} "
                      f"samples to match global_batch_size={self.args.global_batch_size}")
            elif self.data_source.buffer_enabled and len(train_data_samples) < self.args.global_batch_size:
                # This shouldn't happen if our sampling logic is correct, but handle defensively
                print(f"[WARNING] Training data has {len(train_data_samples)} samples, "
                      f"less than global_batch_size={self.args.global_batch_size}. "
                      f"This may cause training issues.")


            # === Step 6: Convert to training format ===
            # IMPORTANT: Verify all samples have valid rewards before conversion
            none_reward_indices = [i for i, s in enumerate(train_data_samples) if s.reward is None]
            if len(none_reward_indices) > 0:
                print(f"[CRITICAL ERROR] Found {len(none_reward_indices)}/{len(train_data_samples)} samples with None rewards before training!")
                print(f"[CRITICAL ERROR] Indices: {none_reward_indices[:10]}{'...' if len(none_reward_indices) > 10 else ''}")
                # Force set to 0 to prevent crash
                for idx in none_reward_indices:
                    print(f"[CRITICAL ERROR] Forcing reward=1e-8 for sample {idx} {train_data_samples[idx]}")
                    train_data_samples[idx].reward = 1e-8

            train_data = self._convert_samples_to_train_data(train_data_samples)

            # === Step 7: Record generated batch in staleness controller ===
            if self.staleness_controller is not None:
                num_samples = len(train_data["tokens"])
                self.staleness_controller.on_generation_completed(num_samples)

            return Box(ray.put(train_data))
        finally:
            if monitor_started:
                self._health_monitor.stop()
                self.num_new_engines = init_rollout_engines(self.args, self.pg, self.all_rollout_engines)
            else:
                self.num_new_engines = 0

    def sample_training_data(self, rollout_id, train_iter):
        """
        Sample training data from buffer without generating new rollout data.

        This method is used for multi-train-per-rollout scenarios where we want to
        train multiple times on the same rollout's buffer without generating new data.

        Args:
            rollout_id: The rollout ID (for logging purposes)
            train_iter: The training iteration within this rollout (0-indexed)

        Returns:
            Box containing ray.put(train_data), or None if buffer is exhausted
        """
        # === Only works in buffer mode ===
        if not (hasattr(self.data_source, 'buffer_enabled') and self.data_source.buffer_enabled):
            raise RuntimeError(
                "sample_training_data() only works in off-policy buffer mode. "
                "For on-policy mode, use generate() instead."
            )

        print(f"[Multi-Train] Rollout {rollout_id}, Train Iteration {train_iter + 1}")

        try:
            # === Step 1: Sample data from buffer (NO rollout generation) ===
            import math
            num_groups_needed = math.ceil(self.args.global_batch_size / self.args.n_samples_per_prompt)

            print(f"[Multi-Train] Sampling {num_groups_needed} groups from buffer "
                  f"(global_batch_size={self.args.global_batch_size}, "
                  f"n_samples_per_prompt={self.args.n_samples_per_prompt}, "
                  f"total_samples={num_groups_needed * self.args.n_samples_per_prompt})")

            train_data_samples = self.data_source.get_training_samples(
                num_samples=num_groups_needed
            )

            # === Step 1.5: Log buffer stats after sampling (just like in generate()) ===
            buffer_stats = self.data_source.get_buffer_stats()
            if buffer_stats["enabled"]:
                wandb_metrics = {
                    "buffer/size": buffer_stats["buffer_size"],
                    "buffer/utilization": buffer_stats["buffer_utilization"],
                    "rollout_id": rollout_id,
                    "train_iter": train_iter,
                }

                # Policy version tracking
                if "current_policy_version" in buffer_stats:
                    wandb_metrics["buffer/current_policy_version"] = buffer_stats["current_policy_version"]

                if "min_policy_version" in buffer_stats:
                    wandb_metrics["buffer/min_policy_version"] = buffer_stats["min_policy_version"]
                    wandb_metrics["buffer/max_policy_version"] = buffer_stats["max_policy_version"]
                    wandb_metrics["buffer/avg_policy_version"] = buffer_stats["avg_policy_version"]

                # Staleness metrics
                if "min_staleness" in buffer_stats:
                    wandb_metrics["buffer/min_staleness"] = buffer_stats["min_staleness"]
                    wandb_metrics["buffer/max_staleness"] = buffer_stats["max_staleness"]
                    wandb_metrics["buffer/avg_staleness"] = buffer_stats["avg_staleness"]

                # Sample reuse metrics
                if "avg_reuse_count" in buffer_stats:
                    wandb_metrics["buffer/avg_reuse_count"] = buffer_stats["avg_reuse_count"]
                    wandb_metrics["buffer/max_reuse_count_observed"] = buffer_stats["max_reuse_count_observed"]

                # Sampling strategy metrics
                if "strategy_strategy" in buffer_stats:
                    wandb_metrics["buffer/strategy"] = buffer_stats["strategy_strategy"]

                # Priority sampling specific metrics
                if "strategy_priority_metric" in buffer_stats:
                    wandb_metrics["buffer/priority_metric"] = buffer_stats["strategy_priority_metric"]
                    wandb_metrics["buffer/priority_weight"] = buffer_stats["strategy_priority_weight"]
                    wandb_metrics["buffer/staleness_penalty"] = buffer_stats["strategy_staleness_penalty"]

                # Base score statistics (raw)
                if "strategy_base_score_mean" in buffer_stats:
                    wandb_metrics["buffer/base_score_mean"] = buffer_stats["strategy_base_score_mean"]
                    wandb_metrics["buffer/base_score_min"] = buffer_stats["strategy_base_score_min"]
                    wandb_metrics["buffer/base_score_max"] = buffer_stats["strategy_base_score_max"]

                # Staleness statistics (raw)
                if "strategy_staleness_mean" in buffer_stats:
                    wandb_metrics["buffer/staleness_stat_mean"] = buffer_stats["strategy_staleness_mean"]
                    wandb_metrics["buffer/staleness_stat_min"] = buffer_stats["strategy_staleness_min"]
                    wandb_metrics["buffer/staleness_stat_max"] = buffer_stats["strategy_staleness_max"]

                # Latest sampling round statistics
                if "strategy_latest_base_score_raw_mean" in buffer_stats:
                    wandb_metrics["buffer/latest_base_score_raw_mean"] = buffer_stats["strategy_latest_base_score_raw_mean"]
                    wandb_metrics["buffer/latest_base_score_raw_std"] = buffer_stats["strategy_latest_base_score_raw_std"]
                    wandb_metrics["buffer/latest_base_score_normalized_mean"] = buffer_stats["strategy_latest_base_score_normalized_mean"]

                if "strategy_latest_staleness_raw_mean" in buffer_stats:
                    wandb_metrics["buffer/latest_staleness_raw_mean"] = buffer_stats["strategy_latest_staleness_raw_mean"]
                    wandb_metrics["buffer/latest_staleness_raw_std"] = buffer_stats["strategy_latest_staleness_raw_std"]
                    wandb_metrics["buffer/latest_staleness_normalized_mean"] = buffer_stats["strategy_latest_staleness_normalized_mean"]

                if "strategy_latest_final_score_mean" in buffer_stats:
                    wandb_metrics["buffer/latest_final_score_mean"] = buffer_stats["strategy_latest_final_score_mean"]
                    wandb_metrics["buffer/latest_final_score_std"] = buffer_stats["strategy_latest_final_score_std"]
                    wandb_metrics["buffer/latest_final_score_min"] = buffer_stats["strategy_latest_final_score_min"]
                    wandb_metrics["buffer/latest_final_score_max"] = buffer_stats["strategy_latest_final_score_max"]

                # Normalization configuration
                if "strategy_normalize_scores" in buffer_stats:
                    wandb_metrics["buffer/normalize_scores"] = 1 if buffer_stats["strategy_normalize_scores"] else 0
                    if buffer_stats.get("strategy_normalization_method"):
                        wandb_metrics["buffer/normalization_method"] = buffer_stats["strategy_normalization_method"]

                # Print to console for debugging
                print(f"[Multi-Train] Sampled {len(train_data_samples)} groups from buffer.")
                print(f"[Buffer Metrics] {wandb_metrics}")

                # Log to WandB
                if wandb.run is not None:
                    wandb.log(wandb_metrics)

            # If buffer is empty/exhausted, skip this training iteration
            if len(train_data_samples) == 0:
                print(f"[Multi-Train] Buffer exhausted, no samples available for training iteration {train_iter + 1}.")
                print(f"[Multi-Train] Skipping remaining training iterations.")

                # Log exhaustion event to WandB
                if wandb.run is not None:
                    wandb.log({
                        "buffer/exhausted": 1,
                        "rollout_id": rollout_id,
                        "train_iter": train_iter,
                    })

                return None

            # === Step 2: Defensive checks (reuse from generate()) ===
            incomplete_groups = []
            for i, group in enumerate(train_data_samples):
                for j, sample in enumerate(group):
                    if sample.status == Sample.Status.PENDING and (not hasattr(sample, 'response') or not sample.response):
                        incomplete_groups.append((i, j, sample))

            if len(incomplete_groups) > 0:
                print(f"[CRITICAL WARNING] Found {len(incomplete_groups)} incomplete samples from buffer!")
                for i, j, sample in incomplete_groups[:5]:
                    print(f"  Group {i}, Sample {j}: "
                          f"group_index={getattr(sample, 'group_index', '?')}, "
                          f"index={getattr(sample, 'index', '?')}, "
                          f"status={sample.status}, "
                          f"has_response={hasattr(sample, 'response') and bool(sample.response)}")
                raise RuntimeError(
                    f"Buffer integrity error: {len(incomplete_groups)} incomplete samples detected!"
                )

            # === Step 3: Flatten grouped samples ===
            if len(train_data_samples) > 0 and isinstance(train_data_samples[0], list):
                flattened_samples = []
                for group in train_data_samples:
                    flattened_samples.extend(group)
                train_data_samples = flattened_samples

            # === Step 4: Trim to exact global_batch_size ===
            if len(train_data_samples) > self.args.global_batch_size:
                original_len = len(train_data_samples)
                train_data_samples = train_data_samples[:self.args.global_batch_size]
                print(f"[Multi-Train] Trimmed training data from {original_len} to {len(train_data_samples)} samples")
            elif len(train_data_samples) < self.args.global_batch_size:
                print(f"[WARNING] Training data has {len(train_data_samples)} samples, "
                      f"less than global_batch_size={self.args.global_batch_size}")

            # === Step 5: Verify rewards ===
            none_reward_indices = [i for i, s in enumerate(train_data_samples) if s.reward is None]
            if len(none_reward_indices) > 0:
                print(f"[CRITICAL ERROR] Found {len(none_reward_indices)}/{len(train_data_samples)} samples with None rewards!")
                for idx in none_reward_indices[:10]:
                    print(f"[CRITICAL ERROR] Forcing reward=1e-8 for sample {idx}")
                    train_data_samples[idx].reward = 1e-8

            # === Step 6: Convert to training format ===
            train_data = self._convert_samples_to_train_data(train_data_samples)

            # === Step 7: Record batch in staleness controller ===
            if self.staleness_controller is not None:
                num_samples = len(train_data["tokens"])
                self.staleness_controller.on_generation_completed(num_samples)

            return Box(ray.put(train_data))

        except Exception as e:
            print(f"[Multi-Train] Error in train iteration {train_iter + 1}: {e}")
            raise

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        # TODO: add fault tolerance to eval
        data = call_rollout_fn(
            self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True
        ).data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        metrics = _log_eval_rollout_data(rollout_id, self.args, data)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        return ray.get([engine.release_memory_occupation.remote() for engine in self.rollout_engines])

    def onload(self, tags: List[str] = None):
        return ray.get([engine.resume_memory_occupation.remote(tags=tags) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                print(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = sum(data, [])

            # === FIX 1: Skip trim in off-policy buffer mode ===
            # In off-policy mode, rollout generates partial batches that go to buffer
            # Training samples from buffer to reach global_batch_size
            # Only trim in on-policy mode where rollout data is used directly for training
            buffer_enabled = hasattr(self.data_source, 'buffer_enabled') and self.data_source.buffer_enabled

            if not buffer_enabled and len(data) % self.args.global_batch_size != 0:
                trim_len = (len(data) // self.args.global_batch_size) * self.args.global_batch_size
                origin_data_length = len(data)
                data = data[:trim_len]
                print(f"[On-Policy] Trimming samples from {origin_data_length} to {trim_len} to match global_batch_size={self.args.global_batch_size}")
            elif buffer_enabled:
                # Off-policy mode: keep all generated samples
                print(f"[Off-Policy] Generated {len(data)} samples (rollout_batch_size={self.args.rollout_batch_size}, "
                      f"n_samples_per_prompt={self.args.n_samples_per_prompt}). "
                      f"Training will sample {self.args.global_batch_size} from buffer.")

        # === Tag samples with current policy version (ENHANCED) ===
        # IMPORTANT: Only set policy_version for NEW samples (from dataset)
        # For samples from buffer (re-generation), they already have policy_version
        # and we should NOT overwrite it, otherwise we lose track of when they were generated

        untagged_count = 0
        retagged_count = 0
        invalid_count = 0

        for sample in data:
            # Case 1: Sample has no policy_version attribute or it's None
            if not hasattr(sample, 'policy_version') or sample.policy_version is None:
                sample.policy_version = self.current_policy_version
                untagged_count += 1

            # Case 2: Sample has an INVALID policy_version (defensive check)
            elif not isinstance(sample.policy_version, int) or sample.policy_version < 0:
                print(f"[WARNING] Sample has invalid policy_version={sample.policy_version}, "
                      f"retagging with current version {self.current_policy_version}")
                sample.policy_version = self.current_policy_version
                invalid_count += 1

            # Case 3: Sample already has a valid policy_version
            else:
                # This is expected for samples from buffer (shouldn't happen in generate())
                # Log if version is much older than current (potential issue)
                staleness = self.current_policy_version - sample.policy_version
                if staleness > 10:  # Threshold for warning
                    print(f"[WARNING] Sample with very old policy_version={sample.policy_version} "
                          f"detected (current={self.current_policy_version}, staleness={staleness}). "
                          f"This may indicate a buffer issue.")

        # Log summary for monitoring
        if untagged_count > 0:
            print(f"[Policy Version] Tagged {untagged_count} new samples with version {self.current_policy_version}")
        if invalid_count > 0:
            print(f"[Policy Version] Fixed {invalid_count} samples with invalid policy_version")

        return data, metrics

    def on_policy_update(self):
        """
        Called after training completes to increment policy version.

        CRITICAL: This method must be called after EVERY training step to ensure
        policy_version stays in sync with actual policy updates.
        """
        old_version = self.current_policy_version
        self.current_policy_version += 1
        new_version = self.current_policy_version

        print(f"[Off-Policy Tracking] Policy version updated: {old_version} -> {new_version}")

        # === Update data source policy version (MANDATORY) ===
        # This MUST succeed to maintain consistency
        if hasattr(self.data_source, 'update_policy_version'):
            self.data_source.update_policy_version(new_version)

            # === VERIFICATION: Check if update succeeded ===
            if hasattr(self.data_source, 'current_policy_version'):
                actual_version = self.data_source.current_policy_version
                if actual_version != new_version:
                    # CRITICAL ERROR: Version mismatch detected
                    raise RuntimeError(
                        f"[CRITICAL] Policy version sync failed! "
                        f"RolloutManager.current_policy_version={new_version}, "
                        f"DataSource.current_policy_version={actual_version}. "
                        f"This will cause incorrect staleness calculation!"
                    )
                print(f"[Off-Policy Tracking] DataSource policy version verified: {actual_version}")
        else:
            # CRITICAL WARNING: DataSource doesn't support policy version tracking
            print(f"[WARNING] DataSource does not implement update_policy_version()! "
                  f"Staleness-aware sampling will NOT work correctly. "
                  f"Please ensure your DataSource class inherits from RolloutDataSourceWithBuffer.")

        # === Update staleness controller ===
        if self.staleness_controller is not None:
            self.staleness_controller.on_training_step()
            print(
                f"[Staleness Control] Updated - num_generated={self.staleness_controller.num_generated}, "
                f"policy_version={self.staleness_controller.current_policy_version}"
            )

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            print(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: Union[list[Sample], list[list[Sample]]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]

        # === NEW: Handle None rewards robustly ===
        # Check for None values and provide meaningful default
        none_count = sum(1 for r in raw_rewards if r is None)

        if none_count > 0:
            # Get valid rewards for statistics
            valid_rewards = [r for r in raw_rewards if r is not None]

            # Determine fallback value based on valid rewards
            if len(valid_rewards) > 0:
                # Use mean of valid rewards as fallback (more reasonable than 0)
                fallback_value = sum(valid_rewards) / len(valid_rewards)
                print(f"[WARNING] Found {none_count}/{len(raw_rewards)} samples with None rewards. "
                      f"Using mean of valid rewards ({fallback_value:.4f}) as fallback.")
            else:
                # All rewards are None - use 0.0 and warn loudly
                # fallback_value = 0.0
                fallback_value = 1e-8
                print(f"[ERROR] ALL {len(raw_rewards)} samples have None rewards! "
                      f"This indicates a serious issue with the reward function. "
                      f"Using fallback value 1e-8, but you should investigate immediately.")

                # Optionally log to wandb if available
                if wandb.run is not None:
                    wandb.log({
                        "error/all_rewards_none": 1,
                        "error/none_reward_count": none_count,
                    })

            # Replace None with fallback value
            raw_rewards_clean = [r if r is not None else fallback_value for r in raw_rewards]

            # Log statistics to wandb
            if wandb.run is not None:
                wandb.log({
                    "reward/none_count": none_count,
                    "reward/none_ratio": none_count / len(raw_rewards),
                    "reward/fallback_value": fallback_value,
                })
        else:
            raw_rewards_clean = raw_rewards

        # Use cleaned rewards for further processing
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards_clean, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            # Return original raw_rewards (with None) for logging, but use clean rewards for training
            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards_clean

    def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
        """
        Convert inference generated samples to training data.
        """
        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
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
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if "teacher_log_probs" in samples[0].__dict__:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        # === Add policy versions for off-policy tracking ===
        # For off-policy GRPO, policy_version is CRITICAL and must always be present
        is_offpolicy_mode = hasattr(self.args, "loss_type") and self.args.loss_type == "decoupled_policy_loss"

        # Collect all policy versions, handling None values robustly
        policy_versions = []
        none_count = 0
        for sample in samples:
            pv = getattr(sample, 'policy_version', None)
            if pv is None:
                none_count += 1
                pv = 0  # Fallback to version 0
            policy_versions.append(pv)

        # Always add policy_versions for off-policy mode
        if is_offpolicy_mode:
            train_data["policy_versions"] = policy_versions
            if none_count > 0:
                print(f"[WARNING] {none_count}/{len(samples)} samples had None policy_version, using 0 as default")
        elif samples[0].policy_version is not None:
            # For on-policy mode, only add if explicitly set
            train_data["policy_versions"] = policy_versions

        return train_data


def init_rollout_engines(args, pg, all_rollout_engines):
    if args.debug_train_only:
        return 0

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine
    assert len(all_rollout_engines) == num_engines

    pg, reordered_bundle_indices = pg

    RolloutRayActor = ray.remote(SGLangEngine)

    rollout_engines = []
    for i in range(num_engines):
        if all_rollout_engines[i] is not None:
            continue

        num_gpus = 0.2
        num_cpus = num_gpus

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        rollout_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={
                "env_vars": {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}
                | {
                    "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                }
            },
        ).remote(args, rank=i)

        rollout_engines.append((i, rollout_engine))
        all_rollout_engines[i] = rollout_engine

    num_new_engines = len(rollout_engines)

    if num_new_engines == 0:
        return num_new_engines

    if args.rollout_external:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=rollout_engines)
    else:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
            args=args, num_engines=num_engines, rollout_engines=rollout_engines
        )

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in rollout_engines]
    ray.get(init_handles)
    return num_new_engines


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = []
    for rank, _ in rollout_engines:
        [host, port] = args.rollout_external_engine_addrs[rank].split(":")
        addr_and_ports.append(
            dict(
                dist_init_addr=None,
                nccl_port=None,
                host=host,
                port=int(port),
            )
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(*, args, num_engines, rollout_engines):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]

    visited_nodes = set()
    for rank, engine in rollout_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports():
            # use small ports to prevent ephemeral port between 32768 and 65536.
            start_port = 10000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports()

        for i in range(num_engines_on_this_node):
            addr_and_ports[rank + i]["port"] = get_port()
            addr_and_ports[rank + i]["nccl_port"] = get_port()

        if args.rollout_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        print(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports


def _start_router(args):
    """start sgl router and slime router"""
    if args.sglang_router_ip is not None:
        return

    args.sglang_router_ip = get_host_info()[1]
    if args.sglang_router_port is None:
        args.sglang_router_port = find_available_port(random.randint(3000, 4000))

    if args.use_slime_router:
        from slime.router.router import run_router

        router_args = args

    else:
        from sglang_router.launch_router import RouterArgs

        from slime.utils.http_utils import run_router

        router_args = RouterArgs(
            host=args.sglang_router_ip,
            port=args.sglang_router_port,
            balance_abs_threshold=0,
            prometheus_port=find_available_port(random.randint(4000, 5000)),
        )

        if hasattr(router_args, "log_level"):
            router_args.log_level = "warn"

        if hasattr(router_args, "request_timeout_secs"):
            router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    print(f"Router launched at {args.sglang_router_ip}:{args.sglang_router_port}")


def _log_eval_rollout_data(rollout_id, args, data):
    log_dict = {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    print(f"eval {rollout_id}: {log_dict}")

    step = (
        rollout_id
        if not args.wandb_always_use_train_step
        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )
    if args.use_wandb:
        log_dict["eval/step"] = step
        wandb.log(log_dict)

    if args.use_tensorboard:
        from slime.utils.tensorboard_utils import _TensorboardAdapter

        tb = _TensorboardAdapter(args)
        tb.log(data=log_dict, step=step)

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.load_debug_rollout_data:
        return

    # === FIX 2: Handle empty samples (defensive programming) ===
    if len(samples) == 0:
        print(f"[WARNING] Rollout {rollout_id} generated 0 samples. Skipping metrics logging.")
        print(f"[WARNING] This may indicate a configuration issue. Check:")
        print(f"  - rollout_batch_size={args.rollout_batch_size}")
        print(f"  - n_samples_per_prompt={args.n_samples_per_prompt}")
        print(f"  - global_batch_size={args.global_batch_size}")
        print(f"  - Expected: rollout_batch_size * n_samples_per_prompt = {args.rollout_batch_size * args.n_samples_per_prompt}")
        return

    log_dict = {**(rollout_extra_metrics or {})}
    response_lengths = [sample.effective_response_length for sample in samples]
    log_dict["perf/rollout_time"] = rollout_time

    # Safely compute metrics only if we have data
    if len(response_lengths) > 0:
        if args.rollout_num_gpus:
            log_dict["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time

    log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), f"rollout/")

    # Also log reward metrics at the top level for easier access in logs
    reward_metrics = {k: v for k, v in log_dict.items() if "generated_reward" in k}

    # Extract format metrics for console logging when format rewards are enabled
    format_metrics = {k: v for k, v in log_dict.items() if k.startswith("rollout/format/")}

    print(f"perf {rollout_id}: {log_dict}")
    if reward_metrics:
        print(f"rollout {rollout_id} generated rewards: {reward_metrics}")

    # Print format statistics if format rewards are enabled and metrics exist
    if getattr(args, 'enable_format_reward', False) and format_metrics:
        # Format for better readability
        format_summary = {}
        for k, v in format_metrics.items():
            # Remove 'rollout/format/' prefix for cleaner display
            clean_key = k.replace('rollout/format/', '')
            # Round ratio values to 3 decimal places
            if 'ratio' in clean_key:
                format_summary[clean_key] = round(v, 3)
            else:
                format_summary[clean_key] = v

        print(f"rollout {rollout_id} format statistics: {format_summary}")
    step = (
        rollout_id
        if not args.wandb_always_use_train_step
        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )
    if args.use_wandb:
        log_dict["rollout/step"] = step
        wandb.log(log_dict)

    if args.use_tensorboard:
        from slime.utils.tensorboard_utils import _TensorboardAdapter

        tb = _TensorboardAdapter(args)
        tb.log(data=log_dict, step=step)


def _compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), f"response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()

    # Add reward statistics for newly generated samples
    log_dict |= _compute_reward_metrics(args, samples)

    # Add format reward statistics when format rewards are enabled
    log_dict |= _compute_format_reward_metrics(args, samples)

    return log_dict


def _compute_zero_std_metrics(args, all_samples: List[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: List[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: List[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["rollout/spec_accept_rate"] = (
        sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    )
    metrics["rollout/spec_accept_length"] = (
        sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    )
    return metrics


def _compute_reward_cat_metrics(args, all_samples: List[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}


def _compute_reward_metrics(args, all_samples: List[Sample]):
    """
    Compute reward statistics from newly generated samples during rollout.

    This records the TRUE rewards produced by the reward model during rollout,
    BEFORE any normalization or post-processing is applied.

    Returns:
        dict: Metrics including:
            - generated_reward/mean: Mean of raw rewards from reward model
            - generated_reward/std: Standard deviation of raw rewards
            - generated_reward/min: Minimum raw reward
            - generated_reward/max: Maximum raw reward
            - generated_reward/median: Median raw reward
    """
    # Extract raw rewards from samples
    raw_rewards = []
    for sample in all_samples:
        try:
            reward_value = sample.get_reward_value(args)
            if reward_value is not None:
                raw_rewards.append(reward_value)
        except Exception as e:
            # Handle cases where reward might not be available
            continue

    # If no valid rewards, return empty dict
    if len(raw_rewards) == 0:
        return {}

    # Convert to numpy array for statistics
    rewards_array = np.array(raw_rewards)

    # Compute statistics using the same compute_statistics function
    metrics = dict_add_prefix(compute_statistics(raw_rewards), f"generated_reward/")

    return metrics


def _compute_format_reward_metrics(args, all_samples: List[Sample]):
    """
    Compute format validation statistics when format rewards are enabled.

    Tracks format correctness, answer accuracy, retrieval success, and reward distribution.
    Only active when --enable-format-reward is set.

    Returns:
        dict: Format metrics including:
            - format/valid_format_ratio: Percentage of samples with correct format
            - format/answer_correct_ratio: Percentage of samples with correct answers
            - format/retrieval_success_ratio: Percentage of samples with successful retrieval
            - format/reward_X.X_count: Distribution of reward values
    """
    # Only compute format metrics if format rewards are enabled
    if not getattr(args, 'enable_format_reward', False):
        return {}

    # Collect format validation statistics
    format_stats = {
        'valid_format': 0,
        'invalid_format': 0,
        'answer_correct': 0,
        'retrieval_success': 0,
        'total': 0,
    }

    # Track reward distribution for fine-grained analysis
    reward_distribution = {}

    for sample in all_samples:
        # Check if sample has format validation metadata
        format_validation = sample.metadata.get('format_validation', None)
        if format_validation is None:
            continue

        format_stats['total'] += 1

        # Count format correctness
        if format_validation.get('is_valid_format', False):
            format_stats['valid_format'] += 1
        else:
            format_stats['invalid_format'] += 1

        # Count answer correctness
        if format_validation.get('answer_correct', False):
            format_stats['answer_correct'] += 1

        # Count retrieval success
        if format_validation.get('retrieval_correct', False):
            format_stats['retrieval_success'] += 1

        # Track reward distribution
        reward_value = sample.get_reward_value(args)
        if reward_value is not None:
            # Round to 1 decimal place for grouping
            reward_key = round(reward_value, 1)
            reward_distribution[reward_key] = reward_distribution.get(reward_key, 0) + 1

    # Return empty dict if no format validation data found
    if format_stats['total'] == 0:
        return {}

    # Compute ratios
    metrics = {
        'format/valid_format_ratio': format_stats['valid_format'] / format_stats['total'],
        'format/invalid_format_ratio': format_stats['invalid_format'] / format_stats['total'],
        'format/answer_correct_ratio': format_stats['answer_correct'] / format_stats['total'],
        'format/retrieval_success_ratio': format_stats['retrieval_success'] / format_stats['total'],
    }

    # Add reward distribution counts
    for reward_value, count in reward_distribution.items():
        metrics[f'format/reward_{reward_value}_count'] = count
        metrics[f'format/reward_{reward_value}_ratio'] = count / format_stats['total']

    return metrics

