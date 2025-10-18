import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict


def save_checkpoint(args, iteration, model, optimizer, tokenizer, global_step, config=None):
    """Save a checkpoint using FSDP distributed checkpointing (FSDP v2 only)."""
    checkpoint_dir = os.path.join(args.save, str(iteration))

    if dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    dist.barrier()

    # Get sharded state dicts
    model_state_dict, optimizer_state_dict = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=False)
    )

    if not optimizer_state_dict or not optimizer_state_dict.get("state", {}):
        raise ValueError(f"Optimizer state dictionary is empty for iteration {iteration}")

    use_safetensors = getattr(args, "save_safe_serialization", False)

    model_subdir = os.path.join(checkpoint_dir, "model")
    optimizer_subdir = os.path.join(checkpoint_dir, "optimizer")

    if dist.get_rank() == 0:
        os.makedirs(model_subdir, exist_ok=True)
        os.makedirs(optimizer_subdir, exist_ok=True)

    dist.barrier()

    # Save model (safetensors if enabled, else standard)
    if use_safetensors:
        try:
            from torch.distributed.checkpoint import HuggingFaceStorageWriter

            # --- FIX: Create the fqn_to_index_mapping with the same prefixed keys
            # that will be used by the checkpoint saver.
            prefixed_fqn_mapping = {f"model.{k}": 0 for k in model_state_dict.keys()}

            model_writer = HuggingFaceStorageWriter(path=model_subdir, fqn_to_index_mapping=prefixed_fqn_mapping)
            # Wrap the model_state_dict in a dictionary, which causes the prefixing.
            dist_cp.save(state_dict={"model": model_state_dict}, storage_writer=model_writer)
        except ImportError as e:
            raise ImportError(
                "Safetensors library is required when save_safe_serialization is True, but it is not installed."
            ) from e
    else:
        model_writer = dist_cp.FileSystemWriter(model_subdir)
        dist_cp.save(state_dict={"model": model_state_dict}, storage_writer=model_writer)

    # Save optimizer (always standard format)
    optim_writer = dist_cp.FileSystemWriter(optimizer_subdir)
    dist_cp.save(state_dict={"optim": optimizer_state_dict}, storage_writer=optim_writer)

    # Save tokenizer, training state, and Hugging Face config on rank 0
    if dist.get_rank() == 0:
        tokenizer.save_pretrained(checkpoint_dir)
        if config is not None:
            config.save_pretrained(checkpoint_dir)
        torch.save(
            {"iteration": iteration, "global_step": global_step}, os.path.join(checkpoint_dir, "training_state.pt")
        )

    dist.barrier()


# The rest of the file (_detect_safetensors and load_checkpoint) remains the same
# as in the previous corrected answer.
def _detect_safetensors(checkpoint_path):
    """Detect if checkpoint uses safetensors format by checking for .safetensors files in model subdir."""
    subpath = os.path.join(checkpoint_path, "model")
    if os.path.isdir(subpath):
        files = os.listdir(subpath)
        if any(f.endswith(".safetensors") for f in files):
            return True
    return False


def load_checkpoint(args, model, optimizer):
    """Load a checkpoint using FSDP distributed checkpointing (FSDP v2 only)."""
    checkpoint_path = args.load
    if not checkpoint_path:
        return -1, 0

    if dist.get_rank() == 0 and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} does not exist.")

    # Detect safetensors format for model
    is_safetensors = False
    if dist.get_rank() == 0 and os.path.exists(checkpoint_path):
        is_safetensors = _detect_safetensors(checkpoint_path)

    is_safetensors_t = torch.tensor([is_safetensors], dtype=torch.long, device="cpu")
    dist.broadcast(is_safetensors_t, src=0)
    is_safetensors = bool(is_safetensors_t[0].item())

    # Get empty state dicts with correct structure
    model_state_dict, optimizer_state_dict = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=False)
    )

    model_subdir = os.path.join(checkpoint_path, "model")
    optimizer_subdir = os.path.join(checkpoint_path, "optimizer")

    # Validate subdirs exist
    if dist.get_rank() == 0:
        if not os.path.exists(model_subdir):
            raise FileNotFoundError(f"Model checkpoint subdirectory {model_subdir} does not exist.")
        if not os.path.exists(optimizer_subdir):
            raise FileNotFoundError(f"Optimizer checkpoint subdirectory {optimizer_subdir} does not exist.")

    dist.barrier()

    # Wrap the model_state_dict for loading as well
    # This ensures consistency with the save operation.
    wrapped_model_state_dict = {"model": model_state_dict}

    # Load model
    if is_safetensors:
        try:
            from torch.distributed.checkpoint import HuggingFaceStorageReader

            model_storage_reader = HuggingFaceStorageReader(path=model_subdir)
            dist_cp.load(state_dict=wrapped_model_state_dict, storage_reader=model_storage_reader)
        except ImportError as e:
            raise ImportError(
                "Safetensors library is required to load safetensors checkpoint files, but it is not installed."
            ) from e
    else:
        model_storage_reader = dist_cp.FileSystemReader(model_subdir)
        dist_cp.load(state_dict=wrapped_model_state_dict, storage_reader=model_storage_reader)

    # After loading, model_state_dict is populated in place within the wrapped dict.

    # Load optimizer (always standard format)
    optim_state_dict = {"optim": optimizer_state_dict}
    optim_storage_reader = dist_cp.FileSystemReader(optimizer_subdir)
    dist_cp.load(state_dict=optim_state_dict, storage_reader=optim_storage_reader)
    optimizer_state_dict = optim_state_dict["optim"]

    if not optimizer_state_dict.get("state", {}):
        raise ValueError("Optimizer state dictionary is empty after loading")

    # Apply loaded state
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state_dict,  # Use the original (now populated) state_dict
        optim_state_dict=optimizer_state_dict,
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )

    # Load training state
    loaded_iteration = -1
    global_step = 0

    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        if dist.get_rank() == 0:
            training_state = torch.load(training_state_path, map_location="cpu")
            loaded_iteration = training_state.get("iteration", -1)
            global_step = training_state.get("global_step", 0)

        # Broadcast to all ranks
        state_t = torch.tensor([0, 0], dtype=torch.int64, device="cpu")
        if dist.get_rank() == 0:
            state_t[0] = loaded_iteration
            state_t[1] = global_step
        dist.broadcast(state_t, src=0)
        loaded_iteration = state_t[0].item()
        global_step = state_t[1].item()

    dist.barrier()
    return loaded_iteration, global_step
