import ray

from slime.hooks import Op, hook
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(rollout_id):
        with hook(Op.OFFLOAD_TRAIN, rollout_id):
            if args.offload_train:
                if args.use_critic:
                    critic_model.offload()
                    if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                        actor_model.offload()
                else:
                    actor_model.offload()
            else:
                if args.critic_train_only:
                    critic_model.clear_memory()
                else:
                    actor_model.clear_memory()

    def save(rollout_id):
        with hook(Op.SAVE_MODEL, rollout_id):
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
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

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        with hook(Op.ITERATION, rollout_id):
            if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
                with hook(Op.EVAL, rollout_id):
                    ray.get(rollout_manager.eval.remote(rollout_id))

            with hook(Op.GENERATE, rollout_id):
                rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

            if args.offload_rollout:
                with hook(Op.OFFLOAD_ROLLOUT, rollout_id):
                    ray.get(rollout_manager.offload.remote())

            with hook(Op.TRAIN, rollout_id):
                if args.use_critic:
                    critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
                    if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
                    ray.get(critic_train_handle)
                else:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

            if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                save(rollout_id)

            offload_train(rollout_id)
            if args.offload_rollout:
                with hook(Op.ONLOAD_ROLLOUT_WEIGHTS, rollout_id):
                    ray.get(rollout_manager.onload_weights.remote())
            if not args.critic_train_only:
                with hook(Op.UPDATE_WEIGHTS, rollout_id):
                    actor_model.update_weights()
            if args.offload_rollout:
                with hook(Op.ONLOAD_ROLLOUT_KV, rollout_id):
                    ray.get(rollout_manager.onload_kv.remote())

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                with hook(Op.EVAL, rollout_id):
                    ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
