import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role

        # Allocate the GPUs for actors w/o instantiating them
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }

        if self.args.offload_train and self.args.offload_train_mode == "tms":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        if self.args.use_routing_replay:
            env_vars["ENABLE_ROUTING_REPLAY"] = "1"

        backend = self.args.train_backend
        if backend == "megatron":
            from slime.backends.megatron_utils import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor

        else:
            from slime.backends.fsdp_utils import FSDPTrainRayActor

            actor_impl = FSDPTrainRayActor

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args, role, with_ref=False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        self.args = args
        return [actor.init.remote(args, role, with_ref=with_ref) for actor in self._actor_handlers]

    def async_train(self, rollout_id, rollout_data_ref):
        """Do one rollout training"""
        return [actor.train.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]

    def save_model(self, step_id):
        """Save actor model on rank 0."""
        return ray.get([actor.save_model.remote(step_id) for actor in self._actor_handlers])

    def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

    def onload(self):
        return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

    def offload(self):
        return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    def clear_memory(self):
        return ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

    def connect(self, critic_group):
        return ray.get(
            [
                actor.connect_actor_critic.remote((critic))
                for actor, critic in zip(self._actor_handlers, critic_group._actor_handlers)
            ]
        )

    def set_rollout_manager(self, rollout_manager):
        return ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])
