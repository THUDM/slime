import logging

import ray

from slime.data.checkpoint import save_checkpoint
from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.misc import should_run_periodic_action


def train(args, pgs, rollout_manager, num_rollout_per_epoch, restore_plan):
    release_train = args.release_train

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()
    # Reattached async producers stay paused until restored weights are installed.
    # This startup notification resumes them once, before entering the train loop.
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
    configure_logger()
    init_tracking(args)
    startup = None
    try:
        # Create or reattach rollout resources, then run this training attempt.
        startup = create_rollout_manager(args, restore_plan=restore_plan)
        train(args, startup.placements, startup.manager, startup.num_rollout_per_epoch, startup.restore_plan)
    except BaseException:
        # Failures and interruptions retain serving and replay data for a manual
        # restart. Detach this driver's trainers instead of disposing the session.
        if startup is not None and startup.serving is not None:
            job_id = ray.get_runtime_context().get_job_id()
            try:
                # Let the manager pause new rollout work and release the trainers.
                ray.get(startup.manager.detach_training.remote(job_id))
            except ray.exceptions.RayActorError:
                # The manager may have died while the serving cluster is still alive.
                try:
                    # Release trainers directly through the independent serving owner.
                    ray.get(startup.serving.detach_training.remote(job_id))
                except Exception:
                    # Report cleanup failure without replacing the original error.
                    logging.getLogger(__name__).exception("Failed to detach after rollout manager death")
            except Exception:
                # A failed detach must not mask the original training failure.
                logging.getLogger(__name__).exception("Failed to detach trainer; serving remains available")
        raise
    else:
        # train has disposed resources after successful completion. Remove the
        # detached actors so later jobs cannot attach to this completed session.
        if startup.serving is not None:
            ray.kill(startup.manager, no_restart=True)
            ray.kill(startup.serving, no_restart=True)
    finally:
        # Finish this driver's tracking on success, startup failure, or interruption.
        finish_tracking(args)
