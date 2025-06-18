import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .ppo_actor import RayTrainGroup
from .rollout import RolloutGroup


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_ip, gpu_id = x
    # Sort by node IP number and then by GPU ID
    node_ip_parts = list(map(int, node_ip.split(".")))
    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    pg_reordered_bundle_indices = [bundle_info[0] for bundle_info in sorted(bundle_infos, key=sort_key)]
    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        print(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices


def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""

    if not args.debug_rollout_only:
        print("Create Actor Placement:")
        actor_pg = _create_placement_group(args.actor_num_nodes * args.actor_num_gpus_per_node)

    if args.colocate:
        rollout_pg = actor_pg
    elif args.debug_train_only:
        rollout_pg = None
    else:
        print("Create Rollout Placement:")
        rollout_pg = _create_placement_group(args.rollout_num_gpus)

    # This is hacky, we hope that when doing rollout_only, we can still instantiate the actor model.
    # And we hope that the rollout could donimate the resource allocation.
    if args.debug_rollout_only:
        assert rollout_pg is not None, "Rollout placement group must be created in debug rollout only mode."
        actor_pg = rollout_pg

    return {
        "actor": actor_pg,
        "rollout": rollout_pg,
    }


def allocate_train_group(num_nodes, num_gpus_per_node, pg, debug_rollout_only):
    return RayTrainGroup(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.8,
        debug_rollout_only=debug_rollout_only,
    )


def create_actor_group(args, pg):
    actor_model = allocate_train_group(
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pg,
        debug_rollout_only=args.debug_rollout_only,
    )
    return actor_model


def create_rollout_group(args, pg):
    return RolloutGroup(args, pg)
