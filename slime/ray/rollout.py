import os
import multiprocessing
import random
import time
import math
import wandb
import asyncio

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SglangEngine
from slime.ray.buffer import Buffer
from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import find_available_port, get_host_info, run_router
from slime.utils.misc import ActorStatus
from slime.utils.wandb_utils import init_wandb_common
from .utils import Lock

from tracer import vinit, TracePoint, MemTracePoint


@ray.remote
class RolloutRayActor(RayActor):
    def __init__(self, args, rank: int, global_rank):
        self.args = args
        self.rank = rank
        self.global_rank = global_rank
        self.status = ActorStatus.PENDING
        os.environ["GLOBAL_RANK"] = str(global_rank)
        vinit()

    def init(self, dist_init_addr, port, nccl_port):
        tp = TracePoint(f"task-{self.args.task_id}: rollout actor init", "1")
        tp.begin()
        MemTracePoint.record("before init infer engine")

        # build infer engine
        self.infer_engine = SglangEngine(
            args=self.args,
            rank=self.rank,
            dist_init_addr=dist_init_addr,
            port=port,
            nccl_port=nccl_port,
            task_id = self.args.task_id,
            global_rank = self.global_rank
        )
        MemTracePoint.record("after init infer engine")

        if self.args.offload:
            MemTracePoint.record("before offload engine")
            # offload the engine to the CPU
            self.sleep()
            MemTracePoint.record("after offload engine")

        tp.end()

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self.infer_engine.init_process_group(
            master_address, master_port, rank_offset, world_size, group_name, backend
        )

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        tp = TracePoint(f"task-{self.args.task_id}: update weights from distributed", "1")
        tp.begin()
        MemTracePoint.record("before update weights from distributed")
        
        result = self.infer_engine.update_weights_from_distributed(names, dtypes, shapes, group_name)
        
        MemTracePoint.record("after update weights from distributed")
        tp.end()
        return result

    def update_weights_from_tensor(self, ipc_handles):
        tp = TracePoint(f"task-{self.args.task_id}: update weights from tensor", "1")
        tp.begin()
        MemTracePoint.record("before update weights from tensor")
        
        result = self.infer_engine.update_weights_from_tensor(ipc_handles)
        
        MemTracePoint.record("after update weights from tensor")
        tp.end()
        return result
    
    def update_weights_from_disk(self, model_path):
        tp = TracePoint(f"task-{self.args.task_id}: update weights from disk", "1")
        tp.begin()
        MemTracePoint.record("before update weights from disk")
        
        result = self.infer_engine.update_weights_from_disk(model_path)
        
        MemTracePoint.record("after update weights from disk")
        tp.end()
        return result

    def reset_prefix_cache(self):
        tp = TracePoint(f"task-{self.args.task_id}: reset prefix cache", "1")
        tp.begin()
        MemTracePoint.record("before reset prefix cache")
        
        self.infer_engine.reset_prefix_cache()
        
        MemTracePoint.record("after reset prefix cache")
        tp.end()

    def sleep(self, level=1):
        assert self.status == ActorStatus.ONLOAD or self.status == ActorStatus.PENDING
        tp = TracePoint(f"task-{self.args.task_id}: rollout actor sleep", "1")
        tp.begin()
        MemTracePoint.record("before engine sleep")
        
        self.infer_engine.sleep(level=level)
        
        MemTracePoint.record("after engine sleep")
        tp.end()
        self.status = ActorStatus.OFFLOAD

    def wake_up(self):
        assert self.status == ActorStatus.OFFLOAD
        tp = TracePoint(f"task-{self.args.task_id}: rollout actor wake up", "1")
        tp.begin()
        MemTracePoint.record("before engine wake up")
        # breakpoint()
        
        self.infer_engine.wake_up()
        
        MemTracePoint.record("after engine wake up")
        self.status = ActorStatus.ONLOAD
        tp.end()


    def pause_generation(self):
        tp = TracePoint(f"task-{self.args.task_id}: rollout pause generation", "1")
        tp.begin()
        MemTracePoint.record("before pause generation")
        
        self.infer_engine.pause_generation()
        
        MemTracePoint.record("after pause generation")
        tp.end()

    def continue_generation(self):
        tp = TracePoint(f"task-{self.args.task_id}: rollout continue generation", "1")
        tp.begin()
        MemTracePoint.record("before continue generation")
        
        self.infer_engine.continue_generation()
        
        MemTracePoint.record("after continue generation")
        tp.end()

    def get_weights_by_name(self, name: str):
        self.infer_engine.get_weights_by_name(name)


def create_rollout_engines(args, pg):
    vinit()
    tp = TracePoint(f"task-{args.task_id}: create_rollout_engines", "1")
    tp.begin()
    if args.debug_train_only:
        return []

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, 8)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine

    pg, offset, reordered_bundle_indices = pg

    rollout_engines = []
    for i in range(num_engines):
        num_gpus = 0.2
        num_cpus = num_gpus

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        rollout_engines.append(
            RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(args, rank=i, global_rank=math.floor(i*num_gpu_per_engine+offset))
        )

    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(1, min(8, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine)
    addr_and_ports = [{} for _ in range(num_engines)]
    for rank, engine in enumerate(rollout_engines):
        if rank % num_engines_per_node != 0:
            continue

        def get_addr_and_ports():
            # use small ports to prevent ephemeral port between 32768 and 65536.
            start_port = 10000+args.task_id*1000+rank*100

            def port(consecutive=10):
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

        for i in range(num_engines_per_node):
            addr_and_ports[rank + i]["port"] = get_port()
            addr_and_ports[rank + i]["nccl_port"] = get_port()

        if args.rollout_num_gpus_per_engine > 8:
            num_node_per_engine = args.rollout_num_gpus_per_engine // 8
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_per_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"

    for i in range(num_engines):
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        print(f"Ports for engine {i}: {addr_and_ports[i]}")

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    init_handles = [engine.init.remote(**ports) for engine, ports in zip(rollout_engines, addr_and_ports)]
    print(f"task-{args.task_id}: {addr_and_ports}")
    ray.get(init_handles)
    tp.end()
    return rollout_engines


class RolloutGroup:
    def __init__(self, args, pg):
        self.args = args
        vinit()
        tp = TracePoint(f"task-{self.args.task_id}: init rollout group", "1")
        tp.begin()
        self.start_router()
        self.data_buffer = Buffer.options(
            num_cpus=1,
            num_gpus=0,
        ).remote(args)

        self.all_rollout_engines = create_rollout_engines(args, pg)
        nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // 8)
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        self.rollout_engines = self.all_rollout_engines[::nodes_per_engine]
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
        ).remote()
        tp.end()

    def start_router(self):
        tp = TracePoint(f"task-{self.args.task_id}: start router", "1")
        tp.begin()
        if self.args.sglang_router_ip is not None:
            return

        from sglang_router.launch_router import RouterArgs

        self.args.sglang_router_ip = get_host_info()[1]
        self.args.sglang_router_port = find_available_port(random.randint(3000+1000*self.args.task_id, 4000+1000*self.args.task_id))

        print("0"*100)
        print(f"{self.args.task_id}: {self.args.sglang_router_ip}, {self.args.sglang_router_port}")
        print("0"*100)

        router_args = RouterArgs(
            host=self.args.sglang_router_ip,
            port=self.args.sglang_router_port,
            prometheus_port=29000+self.args.task_id,
            balance_abs_threshold=0,
        )
    
        print("*"*100)
        print(f"promethus port: {router_args.prometheus_port}")
        print("*"*100)

        if hasattr(router_args, "log_level"):
            router_args.log_level = "warn"

        process = multiprocessing.Process(
            target=run_router,
            args=(router_args,),
        )
        process.daemon = True  # Set the process as a daemon
        process.start()
        # Wait 3 seconds
        time.sleep(3)
        assert process.is_alive()
        # If router ip is specified, use the specified launched router
        print(f"SGLang router launched at {self.args.sglang_router_ip}:{self.args.sglang_router_port}")
        tp.end()

    def async_generate(self, rollout_id, evaluation=False):
        return self.data_buffer.generate.remote(rollout_id, evaluation=evaluation)

    def async_reset_prefix_cache(self):
        return [engine.reset_prefix_cache.remote() for engine in self.rollout_engines]

    def async_offload(self):
        return [engine.sleep.remote() for engine in self.rollout_engines]

    def async_onload(self):
        return [engine.wake_up.remote() for engine in self.rollout_engines]
    
    def async_get_weights_by_name(self, name: str):
        return [engine.get_weights_by_name.remote(name) for engine in self.rollout_engines]