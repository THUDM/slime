import logging

import ray

from slime.data.checkpoint import save_checkpoint
from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.ray.training_recovery import create_recoverable_rollout_manager, training_recovery_enabled
from slime.utils.arguments import parse_args
from slime.utils.misc import should_run_periodic_action


def train(args, restore_plan=None):
    configure_logger()
    recoverable = training_recovery_enabled(args)
    pgs = None if recoverable else create_placement_groups(args)
    init_tracking(args)
    rollout_manager = None
    try:
        if recoverable:
            rollout_manager, pgs, num_rollout_per_epoch, restore_plan = create_recoverable_rollout_manager(
                args, restore_plan
            )
        else:
            rollout_manager, num_rollout_per_epoch = create_rollout_manager(
                args, pgs["rollout"], restore_plan=restore_plan
            )
        _train(args, pgs, rollout_manager, num_rollout_per_epoch, restore_plan)
    except BaseException:
        if recoverable and rollout_manager is not None:
            try:
                ray.get(rollout_manager.detach_training.remote(ray.get_runtime_context().get_job_id()))
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to detach trainer; the detached rollout session remains available"
                )
        raise
    else:
        if recoverable:
            ray.kill(rollout_manager, no_restart=True)
    finally:
        finish_tracking(args)


def _train(args, pgs, rollout_manager, num_rollout_per_epoch, restore_plan):
    release_train = args.release_train

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()
    if training_recovery_enabled(args):
        ray.get(rollout_manager.training_ready.remote())

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # Runtime completion releases queue capacity, without claiming that the
        # model/optimizer or this data progress have a durable joint checkpoint.
        ray.get(rollout_manager.training_completed.remote(rollout_id))

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            save_checkpoint(
                args,
                rollout_id,
                actor_model,
                critic_model,
                rollout_manager,
                actor_trains=actor_trains,
                restore_plan=restore_plan,
            )

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        was_paused = ray.get(rollout_manager.pause_rollout_admission.remote())
        actor_model.update_weights()
        # Final evaluation uses the synchronized engines directly. Keep training
        # producers paused when no later rollout will consume their new work.
        if rollout_id + 1 < args.num_rollout:
            ray.get(rollout_manager.resume_rollout_admission.remote(was_paused))

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args, restore_plan = parse_args(return_restore_plan=True)
    train(args, restore_plan)
