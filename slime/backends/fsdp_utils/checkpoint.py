import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict


def save_checkpoint(args, iteration, model, optimizer, tokenizer, global_step, config=None):
    """Save a checkpoint using FSDP distributed checkpointing (FSDP v2 only)."""
    checkpoint_dir = os.path.join(args.save, str(iteration))

    if dist.get_rank() == 0:
        print(f"Saving checkpoint at iteration {iteration} to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    dist.barrier()

    # Get sharded state dicts
    model_state_dict, optimizer_state_dict = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=False)
    )

    # Debug: Check optimizer state
    if dist.get_rank() == 0:
        if not optimizer_state_dict or not optimizer_state_dict.get("state", {}):
            raise ValueError(f"Optimizer state dictionary is empty for iteration {iteration}")

    use_safetensors = getattr(args, "save_safe_serialization", False)

    if use_safetensors:
        try:
            from torch.distributed.checkpoint import HuggingFaceStorageWriter

            # Flatten optimizer for safetensors compatibility
            flattened_optim = _flatten_optimizer_state(optimizer_state_dict)

            if dist.get_rank() == 0:
                print(f"Optimizer state keys: {len(flattened_optim)}")

            # Create FQN mappings for model and optimizer
            model_fqn_to_index = {f"model.{k}": 0 for k in model_state_dict.keys()}
            optim_fqn_to_index = {f"optim.{k}": 0 for k in flattened_optim.keys()}

            # Save model state dict
            if dist.get_rank() == 0:
                print(f"Saving model checkpoint to {checkpoint_dir}")
            model_storage_writer = HuggingFaceStorageWriter(
                path=checkpoint_dir, fqn_to_index_mapping=model_fqn_to_index
            )
            dist_cp.save(state_dict={"model": model_state_dict}, storage_writer=model_storage_writer)

            # Save optimizer state dict
            if dist.get_rank() == 0:
                print(f"Saving optimizer checkpoint to {checkpoint_dir}")
            optim_storage_writer = HuggingFaceStorageWriter(
                path=checkpoint_dir, fqn_to_index_mapping=optim_fqn_to_index
            )
            dist_cp.save(state_dict={"optim": flattened_optim}, storage_writer=optim_storage_writer)

            if dist.get_rank() == 0:
                print("Saved model and optimizer in safetensors format")

        except ImportError as e:
            raise ImportError(
                "Safetensors library is required when save_safe_serialization is True, but it is not installed."
            ) from e
    else:
        # Save model state dict
        if dist.get_rank() == 0:
            print(f"Saving model checkpoint to {os.path.join(checkpoint_dir, 'model')}")
        model_storage_writer = dist_cp.FileSystemWriter(os.path.join(checkpoint_dir, "model"))
        dist_cp.save(state_dict={"model": model_state_dict}, storage_writer=model_storage_writer)

        # Save optimizer state dict
        if dist.get_rank() == 0:
            print(f"Saving optimizer checkpoint to {os.path.join(checkpoint_dir, 'optimizer')}")
        optim_storage_writer = dist_cp.FileSystemWriter(os.path.join(checkpoint_dir, "optimizer"))
        dist_cp.save(state_dict={"optim": optimizer_state_dict}, storage_writer=optim_storage_writer)

    # Save tokenizer, training state, and Hugging Face config on rank 0
    if dist.get_rank() == 0:
        tokenizer.save_pretrained(checkpoint_dir)
        if config is not None:
            config.save_pretrained(checkpoint_dir)
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
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} does not exist.")
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

            # Load model state dict
            model_state_dict = {"model": model_state_dict}
            model_storage_reader = HuggingFaceStorageReader(path=checkpoint_path)
            dist_cp.load(state_dict=model_state_dict, storage_reader=model_storage_reader)

            # Load optimizer state dict
            optim_state_dict = {"optim": {}}
            optim_storage_reader = HuggingFaceStorageReader(path=checkpoint_path)
            dist_cp.load(state_dict=optim_state_dict, storage_reader=optim_storage_reader)
            optimizer_state_dict = _unflatten_optimizer_state(optim_state_dict["optim"])

        except ImportError as e:
            raise ImportError(
                "Safetensors library is required to load safetensors checkpoint files, but it is not installed."
            ) from e
    else:
        # Load model state dict
        model_checkpoint_path = os.path.join(checkpoint_path, "model")
        if dist.get_rank() == 0 and not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint directory {model_checkpoint_path} does not exist.")
        model_state_dict = {"model": model_state_dict}
        model_storage_reader = dist_cp.FileSystemReader(model_checkpoint_path)
        dist_cp.load(state_dict=model_state_dict, storage_reader=model_storage_reader)

        # Load optimizer state dict
        optim_checkpoint_path = os.path.join(checkpoint_path, "optimizer")
        if dist.get_rank() == 0 and not os.path.exists(optim_checkpoint_path):
            raise FileNotFoundError(f"Optimizer checkpoint directory {optim_checkpoint_path} does not exist.")
        optim_state_dict = {"optim": optimizer_state_dict}
        optim_storage_reader = dist_cp.FileSystemReader(optim_checkpoint_path)
        dist_cp.load(state_dict=optim_state_dict, storage_reader=optim_storage_reader)
        optimizer_state_dict = optim_state_dict["optim"]

    # Apply loaded state
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state_dict["model"],
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
