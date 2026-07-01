import logging

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _free_rollout_data_refs(rollout_data_ref):
    """Release Ray object-store copies of rollout data before checkpoint save."""
    if rollout_data_ref is None:
        return

    object_refs = []
    for shard in rollout_data_ref:
        inner = getattr(shard, "inner", None)
        if isinstance(inner, ray.ObjectRef):
            object_refs.append(inner)

    if not object_refs:
        return

    try:
        from ray._private.internal_api import free

        free(object_refs, local_only=False)
    except Exception as exc:
        logger.warning("Failed to free rollout Ray object refs before checkpoint save: %s", exc)


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        will_save = should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout)

        # Start the next rollout early, except before checkpoint save. Saving a large
        # model already creates a high host-memory peak, so avoid holding the next
        # rollout batch in Ray object store at the same time.
        if rollout_id + 1 < args.num_rollout and not will_save:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
        else:
            rollout_data_next_future = None

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if will_save:
            _free_rollout_data_refs(rollout_data_curr_ref)
            rollout_data_curr_ref = None
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.clear_memory()
            if args.use_critic:
                critic_model.clear_memory()

            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

            if rollout_id + 1 < args.num_rollout:
                rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
