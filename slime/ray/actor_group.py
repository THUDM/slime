from typing import Dict, Optional

import ray

from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.megatron_utils import MegatronTrainRayActor


class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int]],
        wandb_run_id: Optional[str] = None,
        num_gpus_per_actor=1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self._wandb_run_id = wandb_run_id

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        # Allocate the GPUs for actors w/o instantiating them
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor, wandb_run_id=wandb_run_id)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor, wandb_run_id: Optional[str]):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices = pg

        TrainRayActor = ray.remote(
            num_gpus=1,
            runtime_env={
                "env_vars": {
                    # because sglang will always set NCCL_CUMEM_ENABLE to 0
                    # we need also set it to 0 to prevent nccl error.
                    "NCCL_CUMEM_ENABLE": "0",
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                }
            },
        )(MegatronTrainRayActor)

        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port, wandb_run_id)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args, role, with_ref=False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        self.args = args
        return [actor.init.remote(args, role, self._wandb_run_id, with_ref=with_ref) for actor in self._actor_handlers]

    def async_init_weight_update_connections(self, rollout):
        """
        Connect rollout engines and actors, e.g. initialize the process group between them
        to update weights after each training stage.
        """
        self.rollout = rollout
        ray.get([actor.set_data_buffer.remote(rollout.data_buffer) for actor in self._actor_handlers])

        return [
            actor.connect_rollout_engines.remote(
                rollout.rollout_engines,
                rollout.rollout_engine_lock,
            )
            for actor in self._actor_handlers
        ]

    def get_rollout_data(self, rollout_id):
        ray.get([actor.get_rollout_data.remote(rollout_id) for actor in self._actor_handlers])

    def async_train(self, rollout_id, rollout_data_ref):
        """Do one rollout training"""
        return [actor.train.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]

    def async_eval(self, rollout_id, rollout_data_ref):
        """Evaluate the model"""
        return [actor.eval.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]

    def async_save_model(self, step_id):
        """Save actor model on rank 0."""
        return [actor.save_model.remote(step_id) for actor in self._actor_handlers]

    def async_update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        return [actor.update_weights.remote() for actor in self._actor_handlers]

    def async_offload(self):
        return [actor.sleep.remote(("model")) for actor in self._actor_handlers]
