import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


def save_checkpoint(args, iteration, model, optimizer, tokenizer, global_step):
    """Save a checkpoint."""
    if dist.get_rank() == 0:
        checkpoint_dir = os.path.join(args.save, str(iteration))

        if os.path.exists(checkpoint_dir):
            if not args.overwrite_checkpoints:
                print(
                    f"WARNING: Checkpoint directory {checkpoint_dir} already exists and --overwrite-checkpoints is not set. "
                    "Skipping saving."
                )
                return
            else:
                print(f"WARNING: Overwriting existing checkpoint directory {checkpoint_dir}.")

        print(f"Saving checkpoint at iteration {iteration} to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()

        model.save_pretrained(
            checkpoint_dir,
            state_dict=state_dict,
            safe_serialization=args.save_safe_serialization,
        )
        tokenizer.save_pretrained(checkpoint_dir)

        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

        training_state = {
            "iteration": iteration,
            "global_step": global_step,
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        print(f"Checkpoint saved to {checkpoint_dir}")

    dist.barrier()


def load_checkpoint(args, model, optimizer):
    """Load a checkpoint into the provided model and optimizer."""
    loaded_rollout_id = -1
    global_step = 0
    checkpoint_path = args.load
    if not checkpoint_path:
        return loaded_rollout_id, global_step

    print(f"Loading checkpoint from {checkpoint_path}")

    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    if os.path.exists(optimizer_path):
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
        optimizer.load_state_dict(optimizer_state)
        print("Loaded optimizer state.")
    else:
        print("Optimizer state not found, skipping.")

    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path)
        loaded_rollout_id = training_state.get("iteration", -1)
        global_step = training_state.get("global_step", 0)
        print(f"Loaded training state: iteration={loaded_rollout_id}, global_step={global_step}")
    else:
        print("Training state not found, skipping.")

    return loaded_rollout_id, global_step
