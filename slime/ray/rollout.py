import multiprocessing
import random
import time

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SglangEngine
from slime.ray.buffer import Buffer
from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import find_available_port, get_host_info, run_router
from .utils import Lock


@ray.remote
class RolloutRayActor(RayActor):
    def __init__(self, args, rank: int):
        self.args = args
        self.rank = rank

    def init(self, dist_init_addr, port, nccl_port):
        # build infer engine
        self.infer_engine = SglangEngine(
            args=self.args,
            rank=self.rank,
            dist_init_addr=dist_init_addr,
            port=port,
            nccl_port=nccl_port,
        )

        if self.args.offload:
            # offload the engine to the CPU
            self.infer_engine.sleep()

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self.infer_engine.init_process_group(
            master_address, master_port, rank_offset, world_size, group_name, backend
        )

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        return self.infer_engine.update_weights_from_distributed(names, dtypes, shapes, group_name)

    def update_weights_from_tensor(self, ipc_handles):
        return self.infer_engine.update_weights_from_tensor(ipc_handles)

    def reset_prefix_cache(self):
        self.infer_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.infer_engine.sleep(level=level)

    def wake_up(self):
        self.infer_engine.wake_up()

    def pause_generation(self):
        self.infer_engine.pause_generation()

    def continue_generation(self):
        self.infer_engine.continue_generation()


def create_rollout_engines(args, pg):
    if args.debug_train_only:
        return []

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.rollout_num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine

    pg, reordered_bundle_indices = pg

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
            ).remote(args, rank=i)
        )

    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.rollout_num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]
    for rank, engine in enumerate(rollout_engines):
        if rank % num_engines_per_node != 0:
            continue

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

        for i in range(num_engines_per_node):
            addr_and_ports[rank + i]["port"] = get_port()
            addr_and_ports[rank + i]["nccl_port"] = get_port()

        if args.rollout_num_gpus_per_engine > args.rollout_num_gpus_per_node:
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.rollout_num_gpus_per_node
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
    ray.get(init_handles)

    return rollout_engines


def _start_router(args):
    if args.sglang_router_ip is not None:
        return

    from sglang_router.launch_router import RouterArgs

    args.sglang_router_ip = get_host_info()[1]
    args.sglang_router_port = find_available_port(random.randint(3000, 4000))

    router_args = RouterArgs(
        host=args.sglang_router_ip,
        port=args.sglang_router_port,
        balance_abs_threshold=0,
    )

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
    print(f"SGLang router launched at {args.sglang_router_ip}:{args.sglang_router_port}")


class RolloutManager:
    def __init__(self, args, pg, wandb_run_id):
        self.args = args
        _start_router(args)
        self.data_buffer = Buffer.options(
            num_cpus=1,
            num_gpus=0,
        ).remote(args, wandb_run_id=wandb_run_id)

        self.all_rollout_engines = create_rollout_engines(args, pg)
        nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.rollout_num_gpus_per_node)
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        self.rollout_engines = self.all_rollout_engines[::nodes_per_engine]
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
        ).remote()

    def async_generate(self, rollout_id, evaluation=False):
        return self.data_buffer.generate.remote(rollout_id, evaluation=evaluation)

    def async_reset_prefix_cache(self):
        return [engine.reset_prefix_cache.remote() for engine in self.rollout_engines]

    def async_offload(self):
        return [engine.sleep.remote() for engine in self.rollout_engines]

    def async_onload(self):
        return [engine.wake_up.remote() for engine in self.rollout_engines]
