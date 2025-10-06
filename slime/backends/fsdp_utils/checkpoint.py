import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict


def save_checkpoint(args, iteration, model, optimizer, tokenizer, global_step):
    """Save a checkpoint using FSDP distributed checkpointing (FSDP v2 only)."""
    checkpoint_dir = os.path.join(args.save, str(iteration))

    if dist.get_rank() == 0:
        if os.path.exists(checkpoint_dir):
            if not args.overwrite_checkpoints:
                print(
                    f"WARNING: Checkpoint {checkpoint_dir} exists. Skipping (use --overwrite-checkpoints to overwrite)."
                )
                dist.barrier()
                return
            print(f"WARNING: Overwriting checkpoint {checkpoint_dir}")

        print(f"Saving checkpoint at iteration {iteration} to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    dist.barrier()

    # Get sharded state dicts
    model_state_dict, optimizer_state_dict = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=False)
    )

    state_dict = {"model": model_state_dict, "optim": optimizer_state_dict}
    use_safetensors = getattr(args, "save_safe_serialization", False)

    if use_safetensors:
        try:
            from torch.distributed.checkpoint import HuggingFaceStorageWriter

            # Flatten optimizer for safetensors compatibility
            flattened_optim = _flatten_optimizer_state(optimizer_state_dict)
            state_dict["optim"] = flattened_optim

            # Create FQN mapping
            fqn_to_index = {f"model.{k}": 0 for k in model_state_dict.keys()}
            fqn_to_index.update({f"optim.{k}": 0 for k in flattened_optim.keys()})

            storage_writer = HuggingFaceStorageWriter(path=checkpoint_dir, fqn_to_index_mapping=fqn_to_index)

            if dist.get_rank() == 0:
                print("Saving in safetensors format")

        except ImportError:
            if dist.get_rank() == 0:
                print("WARNING: safetensors not available, using standard format")
            storage_writer = dist_cp.FileSystemWriter(checkpoint_dir)
    else:
        storage_writer = dist_cp.FileSystemWriter(checkpoint_dir)

    dist_cp.save(state_dict=state_dict, storage_writer=storage_writer)

    # Save tokenizer and training state on rank 0
    if dist.get_rank() == 0:
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {"iteration": iteration, "global_step": global_step}, os.path.join(checkpoint_dir, "training_state.pt")
        )
        print(f"Checkpoint saved to {checkpoint_dir}")

    dist.barrier()


def load_checkpoint(args, model, optimizer):
    """Load a checkpoint using FSDP distributed checkpointing (FSDP v2 only).

    Returns:
        (iteration, global_step) - iteration is -1 if no training state found
    """
    checkpoint_path = args.load
    if not checkpoint_path:
        return -1, 0

    if dist.get_rank() == 0:
        print(f"Loading checkpoint from {checkpoint_path}")

    # Detect safetensors format
    is_safetensors = False
    if dist.get_rank() == 0 and os.path.exists(checkpoint_path):
        is_safetensors = any(f.endswith(".safetensors") for f in os.listdir(checkpoint_path))
        if is_safetensors:
            print("Detected safetensors format")

    is_safetensors_t = torch.tensor([is_safetensors], dtype=torch.long, device="cpu")
    dist.broadcast(is_safetensors_t, src=0)
    is_safetensors = bool(is_safetensors_t[0].item())

    # Get empty state dicts with correct structure
    model_state_dict, optimizer_state_dict = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=False)
    )

    # Load based on format
    if is_safetensors:
        try:
            from torch.distributed.checkpoint import HuggingFaceStorageReader

            state_dict = {"model": model_state_dict, "optim": {}}
            storage_reader = HuggingFaceStorageReader(path=checkpoint_path)
            dist_cp.load(state_dict=state_dict, storage_reader=storage_reader)

            # Unflatten optimizer state
            optimizer_state_dict = _unflatten_optimizer_state(state_dict["optim"])

        except ImportError:
            if dist.get_rank() == 0:
                print("WARNING: safetensors reader not available, using standard reader")
            state_dict = {"model": model_state_dict, "optim": optimizer_state_dict}
            storage_reader = dist_cp.FileSystemReader(checkpoint_path)
            dist_cp.load(state_dict=state_dict, storage_reader=storage_reader)
            optimizer_state_dict = state_dict["optim"]
    else:
        state_dict = {"model": model_state_dict, "optim": optimizer_state_dict}
        storage_reader = dist_cp.FileSystemReader(checkpoint_path)
        dist_cp.load(state_dict=state_dict, storage_reader=storage_reader)
        optimizer_state_dict = state_dict["optim"]

    # Apply loaded state
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state_dict["model"],
        optim_state_dict=optimizer_state_dict,
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )

    if dist.get_rank() == 0:
        print("Loaded model and optimizer state")

    # Load training state
    loaded_iteration = -1
    global_step = 0

    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        if dist.get_rank() == 0:
            training_state = torch.load(training_state_path, map_location="cpu")
            loaded_iteration = training_state.get("iteration", -1)
            global_step = training_state.get("global_step", 0)
            print(f"Loaded training state: iteration={loaded_iteration}, global_step={global_step}")

        # Broadcast to all ranks
        state_t = torch.tensor([0, 0], dtype=torch.int64, device="cpu")
        if dist.get_rank() == 0:
            state_t[0] = loaded_iteration
            state_t[1] = global_step
        dist.broadcast(state_t, src=0)
        loaded_iteration = state_t[0].item()
        global_step = state_t[1].item()
    else:
        if dist.get_rank() == 0:
            print("No training state found - loaded model weights only")

    dist.barrier()
    return loaded_iteration, global_step


def _flatten_optimizer_state(optimizer_state_dict):
    """Flatten optimizer state for safetensors compatibility."""
    flattened = {}

    for param_name, param_states in optimizer_state_dict.get("state", {}).items():
        for state_key, state_value in param_states.items():
            flattened[f"state.{param_name}.{state_key}"] = state_value

    for group_idx, group in enumerate(optimizer_state_dict.get("param_groups", [])):
        for key, value in group.items():
            if isinstance(value, (torch.Tensor, int, float, bool)) or value is None:
                flattened[f"param_groups.{group_idx}.{key}"] = value

    return flattened


def _unflatten_optimizer_state(flattened_dict):
    """Unflatten optimizer state from safetensors format."""
    state = {}
    param_groups = []

    for fqn, value in flattened_dict.items():
        parts = fqn.split(".", 2)  # Split into at most 3 parts

        if parts[0] == "state":
            param_name, state_key = parts[1], parts[2]
            if param_name not in state:
                state[param_name] = {}
            state[param_name][state_key] = value

        elif parts[0] == "param_groups":
            group_idx, key = int(parts[1]), parts[2]
            while len(param_groups) <= group_idx:
                param_groups.append({})
            param_groups[group_idx][key] = value

    return {"state": state, "param_groups": param_groups}
