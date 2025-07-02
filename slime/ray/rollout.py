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

    def init(self, dist_init_addr, port, nccl_port, other_ports, used_ports):
        # build infer engine
        self.infer_engine = SglangEngine(
            args=self.args,
            rank=self.rank,
            dist_init_addr=dist_init_addr,
            port=port,
            nccl_port=nccl_port,
            other_ports=other_ports,
            used_ports=used_ports,
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

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, 8)
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
    num_engines_per_node = max(1, args.rollout_num_gpus_per_engine // 8)
    addr_and_ports = [
        {"port": None, "nccl_port": None, "dist_init_addr": None, "other_ports": None, "used_ports": None}
        for _ in range(num_engines)
    ]
    for rank, engine in enumerate(rollout_engines):
        if rank % num_engines_per_node != 0:
            continue

        num_ports = num_engines_per_node + num_engines_per_node
        if args.rollout_num_gpus_per_engine > 8:
            assert (
                args.rollout_num_gpus_per_engine % 8 == 0
            ), "rollout_num_gpus_per_engine must be a multiple of 8 for multi-node serving."
            num_node_per_engine = args.rollout_num_gpus_per_engine // 8
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                num_ports += 1
                if args.sglang_enable_dp_attention:
                    # if dp_attention is enabled, we need to allocate additional ports for dp_attention
                    # 4 for the dp_attention and args.sglang_dp_size for the dp_attention
                    num_ports += 4 + args.sglang_dp_size
        else:
            num_ports += num_engines_per_node
            if args.sglang_enable_dp_attention:
                # if dp_attention is enabled, we need to allocate additional ports for dp_attention
                # 4 for the dp_attention and args.sglang_dp_size for the dp_attention
                num_ports += (4 + args.sglang_dp_size) * num_engines_per_node

        addr, ports = ray.get(
            engine._get_current_node_ip_and_free_port.remote(
                # use small ports to prevent ephemeral port between 32768 and 65536.
                num_ports=num_ports,
                start_port=random.randint(10000, 20000),
            )
        )

        for i in range(num_engines_per_node):
            addr_and_ports[rank + i]["used_ports"] = ports.copy()

        for i in range(num_engines_per_node):
            addr_and_ports[rank + i]["port"] = ports.pop(0)
            addr_and_ports[rank + i]["nccl_port"] = ports.pop(0)

        if args.rollout_num_gpus_per_engine > 8:
            num_node_per_engine = args.rollout_num_gpus_per_engine // 8
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{addr}:{ports.pop(0)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr

                if args.sglang_enable_dp_attention:
                    other_ports = []
                    for _ in range(4 + args.sglang_dp_size):
                        other_ports.append(ports.pop(0))
                    for i in range(num_node_per_engine):
                        addr_and_ports[rank + i]["other_ports"] = other_ports
        else:
            for i in range(num_engines_per_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{addr}:{ports.pop(0)}"
                if args.sglang_enable_dp_attention:
                    addr_and_ports[rank + i]["other_ports"] = []
                    for _ in range(4 + args.sglang_dp_size):
                        addr_and_ports[rank + i]["other_ports"].append(ports.pop(0))

        assert len(ports) == 0

    for i in range(num_engines):
        assert addr_and_ports[i]["port"] is not None, f"Engine {i} port is not set."
        assert addr_and_ports[i]["nccl_port"] is not None, f"Engine {i} port is not set."
        assert addr_and_ports[i]["dist_init_addr"] is not None, f"Engine {i} dist_init_addr is not set."
        if args.sglang_enable_dp_attention:
            assert addr_and_ports[i]["other_ports"] is not None, f"Engine {i} other_ports is not set."
            assert (
                len(addr_and_ports[i]["other_ports"]) == 4 + args.sglang_dp_size
            ), f"Engine {i} other_ports should have 4 + sglang_dp_size ports, got {len(addr_and_ports[i]['other_ports'])}."
        print(f"Ports for engine {i}: {addr_and_ports[i]}")

    # don't ray.get here to overlap train actor init with rollout engine init.
    init_handles = [engine.init.remote(**ports) for engine, ports in zip(rollout_engines, addr_and_ports)]
    if args.offload:
        ray.get(init_handles)

    return rollout_engines


class RolloutGroup:
    def __init__(self, args, pg):
        self.args = args
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

    def start_router(self):
        if self.args.sglang_router_ip is not None:
            return

        from sglang_router.launch_router import RouterArgs

        self.args.sglang_router_ip = get_host_info()[1]
        self.args.sglang_router_port = find_available_port(random.randint(3000, 4000))

        router_args = RouterArgs(
            host=self.args.sglang_router_ip,
            port=self.args.sglang_router_port,
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
        print(f"SGLang router launched at {self.args.sglang_router_ip}:{self.args.sglang_router_port}")

    def async_generate(self, rollout_id, evaluation=False):
        return self.data_buffer.generate.remote(rollout_id, evaluation=evaluation)

    def async_reset_prefix_cache(self):
        return [engine.reset_prefix_cache.remote() for engine in self.rollout_engines]

    def async_offload(self):
        return [engine.sleep.remote() for engine in self.rollout_engines]

    def async_onload(self):
        return [engine.wake_up.remote() for engine in self.rollout_engines]
