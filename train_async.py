import time

import ray

from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.misc import should_run_periodic_action
from slime.utils.rs_refill import run_rs_batch_refill


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    release_train = args.release_train
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

        if getattr(args, "rs_batch_refill", False):
            if rollout_data_curr_ref != rollout_id:
                raise RuntimeError(
                    f"RS candidate handle mismatch: expected rollout_id={rollout_id}, got {rollout_data_curr_ref}"
                )
            rollout_data_curr_ref = run_rs_batch_refill(
                actor_model,
                rollout_manager,
                rollout_id,
                resolve=ray.get,
                clock=time.perf_counter,
                rpc_timeout_seconds=args.rs_refill_rpc_timeout_seconds,
            )

        save_this_step = release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        # The data-source checkpoint must not include a prefetched batch that
        # is absent from the corresponding model checkpoint.
        defer_next_rollout = getattr(args, "rs_batch_refill", False) and save_this_step

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout and not defer_next_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if save_this_step:
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if release_train or (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if defer_next_rollout and rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
