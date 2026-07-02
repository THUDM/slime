import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    def release_train_models():
        if actor_model.is_started:
            actor_model.shutdown()
        if args.use_critic and critic_model.is_started:
            critic_model.shutdown()

    def ensure_train_model(model):
        if not model.is_started:
            model.restart()

    def use_latest_saved_checkpoint(model):
        model.args.load = model.args.save
        model.args.ckpt_step = None
        model.args.no_load_optim = False
        model.args.no_load_rng = False
        model.args.finetune = False

    def actor_hf_path(rollout_id):
        assert actor_model.args.save_hf is not None, "--reload-train-from-disk requires actor --save-hf."
        return actor_model.args.save_hf.format(rollout_id=rollout_id)

    if args.reload_train_from_disk:
        release_train_models()

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

    def save(rollout_id, force_sync=False):
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        actor_saved = False
        if actor_trains_this_step:
            actor_model.save_model(
                rollout_id,
                force_sync=force_sync or rollout_id == args.num_rollout - 1,
            )
            actor_saved = True
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=force_sync or rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))
        return actor_saved

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        if args.reload_train_from_disk:
            if args.use_critic:
                ensure_train_model(critic_model)
            if actor_trains_this_step:
                ensure_train_model(actor_model)

        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        actor_saved = False
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            actor_saved = save(rollout_id, force_sync=args.reload_train_from_disk)

        if args.reload_train_from_disk:
            if actor_trains_this_step:
                assert (
                    actor_saved
                ), "--reload-train-from-disk requires actor checkpoint/HF save after every actor train."
                use_latest_saved_checkpoint(actor_model)
            if args.use_critic:
                use_latest_saved_checkpoint(critic_model)
            release_train_models()
        else:
            offload_train(actor_trains_this_step)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        if args.reload_train_from_disk:
            if actor_trains_this_step:
                ray.get(
                    rollout_manager.update_updatable_weights_from_disk.remote(
                        actor_hf_path(rollout_id),
                        weight_version=str(rollout_id + 1),
                    )
                )
        else:
            actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
