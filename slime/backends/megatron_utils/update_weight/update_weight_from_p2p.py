"""P2P full-weight updates for disaggregated Megatron and SGLang runtimes."""

import logging
import re
import threading
import time

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.backends.sglang_utils.p2p import encode_manifest
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.p2p import BufferPool, Publisher
from slime.utils.reloadable_process_group import ReloadableProcessGroup
from slime.utils.weight_sync import should_flush_cache

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers

logger = logging.getLogger(__name__)
_LAYER = re.compile(r"^(.*?(?:decoder|transformer|mtp)\.layers\.\d+)\.")


def iter_owned_hf_tensors(args, model, model_name, quantization_config):
    """Convert each weight once, retaining training-side TP collectives.

    Experts stay on their EP owner. Other layers are balanced over DP/CP
    replicas; keeping a layer together also keeps paired MLA projections on
    one source. SGLang's native loader handles inference-side partitioning.
    """
    params = list(named_params_and_buffers(args, model))
    dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
    load = [0] * mpu.get_data_parallel_world_size(with_context_parallel=True)
    owners = {}
    for name, param in params:
        if ".experts." in name:
            continue
        match = _LAYER.match(name)
        key = match.group(1) if match else name
        if key not in owners:
            owners[key] = min(range(len(load)), key=load.__getitem__)
        load[owners[key]] += param.numel() * param.element_size()
    for name, param in params:
        expert = ".experts." in name
        if expert:
            if mpu.get_expert_data_parallel_rank() != 0:
                continue
        else:
            match = _LAYER.match(name)
            if owners[match.group(1) if match else name] != dp_rank:
                continue
        param = all_gather_param(name, param)
        rank = mpu.get_expert_tensor_parallel_rank() if expert else mpu.get_tensor_model_parallel_rank()
        if rank == 0:
            yield convert_to_hf(args, model_name, name, param, quantization_config)


class UpdateWeightFromP2P:
    def __init__(self, args, model, weights_getter, *, model_name, quantization_config):
        self.args, self.model = args, model
        self.model_name, self.quantization_config = model_name, quantization_config
        self.weight_version = 0
        self.update_weight_metrics = {}
        self._pool = None
        self._failed = False

    def pop_metrics(self):
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    def connect_rollout_engines(
        self,
        rollout_engines,
        rollout_engine_lock,
        engine_gpu_counts=None,
        engine_gpu_offsets=None,
        engine_parallel_configs=None,
    ):
        counts = engine_gpu_counts or [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if not rollout_engines or len(counts) != len(rollout_engines) or any(count <= 0 for count in counts):
            raise ValueError("P2P requires rollout engines with positive GPU counts")
        configs = engine_parallel_configs or [{} for _ in counts]
        if len(configs) != len(counts):
            raise ValueError("P2P engine parallel configs must match engine GPU counts")
        for count, config in zip(counts, configs, strict=True):
            if int(config.get("pp_size", 1)) != 1 or int(config.get("tp_size", count)) != count:
                raise ValueError("The P2P tensor loader requires serving PP=1 and one worker per TP rank")
        self.rollout_engines, self._engine_gpu_counts = list(rollout_engines), list(counts)

    def disconnect_rollout_engines(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _coordinator_call(self, method, **kwargs):
        """Propagate engine failures to all trainers before the next collective."""
        failure = [None]
        if dist.get_rank() == 0:
            try:
                results = ray.get(
                    [getattr(engine, method).remote(**kwargs) for engine in self.rollout_engines],
                    timeout=self.args.update_weight_p2p_timeout,
                )
                _check_engine_results(results)
            except Exception as exc:
                failure[0] = repr(exc)
        dist.broadcast_object_list(failure, src=0, group=get_gloo_group())
        if failure[0]:
            raise RuntimeError(f"P2P {method} failed: {failure[0]}")

    @torch.no_grad()
    def update_weights(self):
        if self._failed:
            raise RuntimeError("A P2P update failed; restart the affected trainers and engines before retrying")
        started = time.perf_counter()
        version = self.weight_version + 1
        flush = should_flush_cache(self.args.flush_cache_interval, version, self.args.update_weight_start_version)
        quantized = self.quantization_config and self.quantization_config["quant_method"] == "compressed-tensors"
        try:
            self._coordinator_call("pause_generation", mode="abort" if flush else "in_place")
            if flush:
                self._coordinator_call("flush_cache")
            if quantized:
                self._coordinator_call(
                    "post_process_weights", restore_weights_before_load=True, post_process_quantization=False
                )
            self._send_weights(version)
            if quantized:
                self._coordinator_call(
                    "post_process_weights", restore_weights_before_load=False, post_process_quantization=True
                )
            self._coordinator_call("set_weight_version", new_version=str(version))
            # All engines have loaded the complete version before any resumes.
            self._coordinator_call("continue_generation")
        except Exception:
            self._failed = True
            # No automatic retry: model weights may already be partially loaded.
            raise
        self.weight_version = version
        self.update_weight_metrics["update_weights/p2p_time"] = time.perf_counter() - started

    def _send_weights(self, version):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        consumers = [f"{engine}:{gpu}" for engine, count in enumerate(self._engine_gpu_counts) for gpu in range(count)]
        capacity = max(1024 * 1024, self.args.update_weight_buffer_size // world_size)
        timeout = self.args.update_weight_p2p_timeout
        publisher, failure = None, None
        groups = []
        try:
            groups = list(
                {
                    group
                    for group in (mpu.get_tensor_model_parallel_group(), mpu.get_expert_tensor_parallel_group())
                    if dist.get_world_size(group=group) > 1
                }
            )
            if groups and not hasattr(dist.distributed_c10d, "_abort_process_group"):
                raise RuntimeError("P2P conversion with TP requires PyTorch's process-group abort API")
            host = ray._private.services.get_node_ip_address()
            if self._pool is None:
                self._pool = BufferPool(host, device="cuda")
            publisher = Publisher(
                host, version=version, consumers=consumers, capacity=capacity, timeout=timeout, pool=self._pool
            )
        except Exception as exc:
            failure = repr(exc)
        states = [None] * world_size
        dist.all_gather_object(states, (failure, publisher.endpoint if publisher else None), group=get_gloo_group())
        if any(error for error, _ in states):
            if publisher:
                publisher.close()
            if self._pool:
                self._pool.trim()
            raise RuntimeError(f"P2P initialization failed: {[error for error, _ in states if error]}")
        endpoints = [endpoint for _, endpoint in states]
        refs = []
        watch = None
        payload_bytes = 0
        try:
            if rank == 0:
                try:
                    for index, (engine, count) in enumerate(
                        zip(self.rollout_engines, self._engine_gpu_counts, strict=True)
                    ):
                        manifests = [
                            encode_manifest(
                                {
                                    "endpoints": endpoints,
                                    "version": str(version),
                                    "consumer": f"{index}:{gpu}",
                                    "timeout": timeout,
                                }
                            )
                            for gpu in range(count)
                        ]
                        refs.append(engine.update_weights_from_p2p.remote(serialized_manifests=manifests))
                except Exception as exc:
                    failure = repr(exc)
            launch = [failure]
            dist.broadcast_object_list(launch, src=0, group=get_gloo_group())
            if launch[0]:
                raise RuntimeError(f"P2P receiver launch failed: {launch[0]}")
            watch = _AbortWatch(
                dist.distributed_c10d._get_default_store(), f"p2p-abort-{endpoints[0]['token']}", groups, refs, timeout
            )
            bucket, size = [], 0
            for unit in iter_owned_hf_tensors(self.args, self.model, self.model_name, self.quantization_config):
                watch.check()
                unit_size = sum((tensor.numel() * tensor.element_size() + 63) // 64 * 64 for _, tensor in unit)
                if bucket and size + unit_size > capacity:
                    publisher.publish(bucket)
                    bucket, size = [], 0
                bucket.extend(unit)
                size += unit_size
                payload_bytes += sum(tensor.numel() * tensor.element_size() for _, tensor in unit)
            if bucket:
                publisher.publish(bucket)
            publisher.finish()
            if rank == 0:
                _check_engine_results(ray.get(refs, timeout=timeout))
        except Exception as exc:
            failure = repr(exc)
            publisher.abort(exc)
            if watch:
                watch.publish(f"trainer rank {rank}: {exc!r}")
                watch.abort()
        finally:
            publisher.close()
            try:
                bucket, unit = [], None
                self._pool.trim()
                torch.cuda.empty_cache()
            except Exception as exc:
                failure = failure or repr(exc)
            if watch:
                watch.close()
                failure = failure or watch.cause()
        failures = [None] * world_size
        dist.all_gather_object(failures, failure, group=get_gloo_group())
        if any(failures):
            raise RuntimeError(f"P2P update failed; engines remain paused: {[error for error in failures if error]}")
        self.update_weight_metrics["update_weights/p2p_source_bytes"] = payload_bytes


def _check_engine_results(results):
    for result in results:
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(result.get("message", "Weight update failed"))


class _AbortWatch:
    """Unblock training TP gathers if a sender or serving worker fails.

    Only training groups are aborted. The data plane never creates a process
    group spanning training and serving runtimes. An aborted update requires
    process restart, so there is no attempt to reuse these communicators.
    """

    def __init__(self, store, key, groups, refs, timeout):
        self.store, self.key = store, key
        self.groups = [group.group if isinstance(group, ReloadableProcessGroup) else group for group in groups]
        self.reason = None
        self._lock = threading.Lock()
        self._aborted = False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._watch, args=(list(refs), timeout), daemon=True)
        self._thread.start()

    def _watch(self, refs, timeout):
        deadline = time.monotonic() + timeout
        while not self._done.wait(0.05):
            try:
                if time.monotonic() >= deadline:
                    self.publish("P2P update timed out")
                if refs:
                    ready, refs = ray.wait(refs, num_returns=len(refs), timeout=0)
                    for ref in ready:
                        try:
                            _check_engine_results([ray.get(ref)])
                        except Exception as exc:
                            self.publish(f"engine: {exc!r}")
                if self.store.check([self.key]):
                    self.reason = self.store.get(self.key).decode()
            except Exception as exc:
                self.reason = f"P2P abort store unreachable: {exc!r}"
            if self.reason is not None:
                self.abort()
                return

    def publish(self, reason):
        self.store.compare_set(self.key, "", reason[:4000])

    def abort(self):
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
        # Aborting one NCCL group can wait for another group's kernel, so abort
        # overlapping dense/expert TP groups concurrently.
        threads = [threading.Thread(target=self._abort_group, args=(group,)) for group in self.groups]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    @staticmethod
    def _abort_group(group):
        try:
            dist.distributed_c10d._abort_process_group(group)
        except Exception:
            logger.exception("Failed to abort a training process group during P2P weight update")

    def check(self):
        if self.reason is not None:
            raise RuntimeError(f"P2P update aborted: {self.reason}")

    def close(self):
        self._done.set()
        self._thread.join()

    def cause(self):
        try:
            return self.store.get(self.key).decode() if self.store.check([self.key]) else self.reason
        except Exception:
            return self.reason
