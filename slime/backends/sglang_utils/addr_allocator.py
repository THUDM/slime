"""Rollout engine address allocation and placement layout validation.

Kept free of heavy imports (sglang, torch) so that it is unit-testable and
importable from both the deployment code and tests.
"""

import logging

import ray

logger = logging.getLogger(__name__)


def assert_group_layout_valid(
    *,
    group_config,
    gpus_per_engine: int,
    group_abs_start: int,
    num_gpus_per_node: int,
) -> None:
    """Reject layouts that node-width formulas would silently misplace.

    A group can start in the middle of a physical node because the training
    actor owns the placement-group prefix.  Guard every formula that otherwise
    assumes whole nodes:
      - an engine either fits in one node or tiles whole nodes;
      - engines (or per-node shards of a multi-node engine) occupy consecutive
        placement-group slots, so each one's first slot must stay aligned
        within its node, otherwise it would straddle a node boundary and be
        assigned GPUs that do not exist there.
    """
    assert gpus_per_engine <= num_gpus_per_node or gpus_per_engine % num_gpus_per_node == 0, (
        f"group '{group_config.worker_type}' wants num_gpus_per_engine={gpus_per_engine} which neither "
        f"fits in one node of {num_gpus_per_node} gpus nor tiles whole nodes"
    )
    num_gpus_per_engine_on_node = min(gpus_per_engine, num_gpus_per_node)
    assert (
        num_gpus_per_node % num_gpus_per_engine_on_node == 0
        and (group_abs_start % num_gpus_per_node) % num_gpus_per_engine_on_node == 0
    ), (
        f"group '{group_config.worker_type}' would straddle a node boundary: "
        f"group starts at GPU slot {group_abs_start} and each engine takes "
        f"{num_gpus_per_engine_on_node} consecutive GPUs, which does not tile "
        f"the {num_gpus_per_node}-GPU nodes. Align actor/rollout GPU counts "
        f"and num_gpus_per_engine with the node width."
    )
    assert group_config.num_gpus % gpus_per_engine == 0, (
        f"group '{group_config.worker_type}' has num_gpus={group_config.num_gpus} "
        f"which is not divisible by num_gpus_per_engine={gpus_per_engine}"
    )


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    """Allocate rank-local SGLang and distributed-init ports for one group.

    Each engine's host is measured from its Ray actor rather than inferred
    from its rank.  A rollout group can start in the middle of a physical
    node because the training actor owns the placement-group prefix (e.g.
    actor=4 and rollout=12 on two 8-GPU nodes), so a ``local_rank //
    engines_per_node``-style formula would assign engines to the wrong node.
    """
    gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_node_per_engine = max(1, gpus_per_engine // args.num_gpus_per_node)
    addr_and_ports: dict[int, dict] = {}
    # Keyed by the physical host so that different server groups (called
    # sequentially) never race for the same ports on a given host.
    host_port_cursor: dict[str, int] = {}

    engine_hosts = {
        rank: ray.get(engine._get_current_node_ip_and_free_port.remote())[0] for rank, engine in rollout_engines
    }

    def get_port(engine, host, consecutive=1):
        # Use small ports to prevent ephemeral ports between 32768 and 65536.
        # Ray uses ports 10002-19999, so start near 15000.
        start_port = host_port_cursor.get(host, base_port)
        actual_host, port = ray.get(
            engine._get_current_node_ip_and_free_port.remote(
                start_port=start_port,
                consecutive=consecutive,
            )
        )
        if actual_host != host:
            raise RuntimeError(
                f"Rollout engine moved hosts while allocating ports: expected {host}, got {actual_host}."
            )
        host_port_cursor[host] = port + consecutive
        return port

    for rank, engine in rollout_engines:
        host = engine_hosts[rank]
        addr_and_ports[rank] = {
            "host": host,
            "port": get_port(engine, host),
            "nccl_port": get_port(engine, host),
            # Group-relative so multi-node groups after another group with a
            # different nodes_per_engine still align with their dist_init group.
            "node_rank": (rank - rank_offset) % num_node_per_engine,
        }

        if worker_type == "prefill":
            addr_and_ports[rank]["disaggregation_bootstrap_port"] = get_port(engine, host)

    if gpus_per_engine > args.num_gpus_per_node:
        # One engine spans several physical nodes; its dist_init_addr is
        # allocated on the first shard's actual host and shared by the
        # engine's consecutive shard ranks.
        for rank, engine in rollout_engines:
            if (rank - rank_offset) % num_node_per_engine != 0:
                continue
            dist_init_addr = f"{engine_hosts[rank]}:{get_port(engine, engine_hosts[rank], 30 + args.sglang_dp_size)}"
            for shard_rank in range(rank, rank + num_node_per_engine):
                if shard_rank in addr_and_ports:
                    addr_and_ports[shard_rank]["dist_init_addr"] = dist_init_addr
    else:
        for rank, engine in rollout_engines:
            host = engine_hosts[rank]
            addr_and_ports[rank]["dist_init_addr"] = f"{host}:{get_port(engine, host, 30 + args.sglang_dp_size)}"

    for rank, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[rank], f"Engine {rank} {key} is not set."
        logger.info(f"Ports for engine {rank}: {addr_and_ports[rank]}")

    return addr_and_ports, host_port_cursor
