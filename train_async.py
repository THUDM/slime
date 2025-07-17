import ray

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_group
from slime.utils.arguments import parse_args


def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    # allocate the GPUs
    pgs = create_placement_groups(args)

    actor_model = create_actor_group(args, pgs["actor"])

    # create the rollout generator, with sglang engines inside.
    rollout_generator = create_rollout_group(args, pgs["rollout"])

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_generator.data_buffer.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    # Note that we initialize it earlier as megatron ckpt loading may have really large peak memory usage.
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_generator.data_buffer.load.remote(args.start_rollout_id - 1))

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_generator))

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    # async train loop.
    generation_handles = rollout_generator.async_generate(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        ray.get(generation_handles)

        # This is a synchronous call to ensure that the rollout data is ready
        actor_model.get_rollout_data(rollout_id)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            generation_handles = rollout_generator.async_generate(rollout_id + 1)

        ray.get(actor_model.async_train(rollout_id, with_data_fetching=False))

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_generator.data_buffer.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            ray.get(generation_handles)
            ray.get(actor_model.async_update_weights())

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_generator.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id))


if __name__ == "__main__":
    args = parse_args()
    train(args)
