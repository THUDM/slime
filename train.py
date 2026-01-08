import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary


def train(args):
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_wandb_primary(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    if args.offload_train and not args.enable_weights_backuper:
        actor_model.onload()
    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()
    if args.offload_train and not args.enable_weights_backuper:
        actor_model.offload()

    if args.offload_rollout:
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def onload_rollout():
        if args.offload_rollout:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_manager.eval.remote(rollout_id))

        # === NEW: Multi-train per rollout support ===
        # Get train_iters_per_rollout parameter (default=1 for backward compatibility)
        train_iters_per_rollout = getattr(args, 'train_iters_per_rollout', 1)
        update_policy_every_iter = getattr(args, 'update_policy_version_every_train_iter', False)

        if train_iters_per_rollout > 1:
            # Check if buffer is enabled (multi-train only works in off-policy mode)
            buffer_enabled = hasattr(args, 'loss_type') and args.loss_type == 'decoupled_policy_loss'
            if not buffer_enabled:
                print(f"[WARNING] train_iters_per_rollout={train_iters_per_rollout} is set but buffer is not enabled.")
                print(f"[WARNING] Multi-train per rollout only works in off-policy mode (loss_type=decoupled_policy_loss).")
                print(f"[WARNING] Falling back to standard single-train behavior.")
                train_iters_per_rollout = 1

        # === Train Iteration Loop ===
        for train_iter in range(train_iters_per_rollout):
            if train_iter == 0:
                # First iteration: generate new rollout data + sample from buffer
                rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
            else:
                # Subsequent iterations: only sample from buffer (no new rollout)
                rollout_data_ref = ray.get(rollout_manager.sample_training_data.remote(rollout_id, train_iter))

            # 🔧 FIX: Handle case where training is skipped due to exhausted buffer
            if rollout_data_ref is None:
                if train_iter == 0:
                    print(f"[Train] Rollout {rollout_id}: Training skipped - buffer exhausted, no samples available.")
                    print(f"[Train] Continuing to next rollout to generate new data...")
                else:
                    print(f"[Multi-Train] Rollout {rollout_id}, iteration {train_iter + 1}: Buffer exhausted.")
                    print(f"[Multi-Train] Completed {train_iter}/{train_iters_per_rollout} training iterations.")
                break  # Exit train iteration loop

            # 🔧 FIX: Offload rollout engine before training to prevent OOM in multi-train mode
            if args.offload_rollout:
                # First iteration: rollout was just generated, need to offload
                # Subsequent iterations: rollout was onloaded in previous iteration for weight update (line 136-139), need to offload again
                ray.get(rollout_manager.offload.remote())

            # === Execute training ===
            if args.use_critic:
                critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
                if rollout_id >= args.num_critic_only_steps:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
                ray.get(critic_train_handle)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

            # === Policy version update logic ===
            if update_policy_every_iter:
                # Update policy version after each training iteration
                ray.get(rollout_manager.on_policy_update.remote())
                if train_iter < train_iters_per_rollout - 1:
                    print(f"[Multi-Train] Policy version updated after iteration {train_iter + 1}/{train_iters_per_rollout}")
            elif train_iter == train_iters_per_rollout - 1:
                # Default: only update after all training iterations complete
                ray.get(rollout_manager.on_policy_update.remote())

            if train_iter < train_iters_per_rollout - 1:
                # Not the last iteration - prepare for next training iteration
                # Update weights before next iteration (important for multi-train)
                if args.enable_weights_backuper:
                    offload_train()
                    onload_rollout()
                    actor_model.update_weights()
                else:
                    actor_model.clear_memory()
                    onload_rollout()
                    actor_model.update_weights()
                    offload_train()

                if args.offload_rollout:
                    if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
                    ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        # === End of train iteration loop ===

        # === Post-training operations (after all train iterations) ===
        # These operations only run once per rollout, not per train iteration

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                actor_model.save_model(rollout_id)
            if args.use_critic:
                critic_model.save_model(rollout_id)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        # Update weights after all train iterations (if not already done in loop)
        if train_iters_per_rollout == 1 or not update_policy_every_iter:
            # Standard single-train or multi-train without per-iteration updates
            if args.enable_weights_backuper:
                offload_train()
                onload_rollout()
                actor_model.update_weights()
            else:
                actor_model.clear_memory()
                onload_rollout()
                actor_model.update_weights()
                offload_train()

            if args.offload_rollout:
                if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                    ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))
        # else: weights were already updated in the last train iteration loop

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
