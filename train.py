import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_group
from slime.ray.registry import register_actor
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary


def train(args):
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    actor_model = create_training_group(args, pgs["actor"], wandb_run_id=wandb_run_id)
    if args.use_critic:
        critic_model = create_training_group(args, pgs["critic"], wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    # TODO: extract this to single function
    rollout_engines, rollout_engine_lock = ray.get(rollout_manager.get_rollout_engines_and_lock.remote())
    for i, rollout_engine in enumerate(rollout_engines):
        register_actor("rollout", i, rollout_engine)
    register_actor("rollout_lock", 0, rollout_engine_lock)
    for i, actor in enumerate(actor_model._actor_handlers):
        register_actor("actor", i, actor)
    if args.use_critic:
        for i, critic in enumerate(critic_model._actor_handlers):
            register_actor("critic", i, critic)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    if args.use_critic:
        critic_init_handle = critic_model.async_init(args, role="critic", with_ref=False)

    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    if args.use_critic:
        ray.get(critic_init_handle)
        ray.get(actor_model.async_connect(critic_model))

    if args.offload:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    if args.offload:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if args.offload:
            if args.use_critic:
                ray.get(critic_model.async_offload())
                if rollout_id >= args.num_critic_only_steps:
                    ray.get(actor_model.async_offload())
            else:
                ray.get(actor_model.async_offload())

            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

        ray.get(actor_model.async_update_weights())

        if args.offload:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))


if __name__ == "__main__":
    args = parse_args()
    train(args)
