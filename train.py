import ray

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary


def train(args):
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    actor_model = create_actor_group(args, pgs["actor"], wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.data_buffer.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_manager.data_buffer.load.remote(args.start_rollout_id - 1))

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_manager))

    if args.offload:
        ray.get(rollout_manager.async_onload())

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            eval_rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id, eval_rollout_data_ref))

        rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id))

        if args.offload:
            ray.get(rollout_manager.async_offload())

        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_manager.data_buffer.save.remote(rollout_id))

        if args.offload:
            ray.get(actor_model.async_offload())
            ray.get(rollout_manager.async_onload())

        ray.get(actor_model.async_update_weights())

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            eval_rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id, eval_rollout_data_ref))


if __name__ == "__main__":
    args = parse_args()
    train(args)
