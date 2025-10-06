import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from packaging import version


def save_checkpoint(args, iteration, model, optimizer, tokenizer, global_step):
    """
    Save a checkpoint using FSDP distributed checkpointing.
    Compatible with both FSDP v1 and v2 (composable fully_shard).
    """
    checkpoint_dir = os.path.join(args.save, str(iteration))

    # Only rank 0 checks/creates directory and prints messages
    if dist.get_rank() == 0:
        if os.path.exists(checkpoint_dir):
            if not args.overwrite_checkpoints:
                print(
                    f"WARNING: Checkpoint directory {checkpoint_dir} already exists and --overwrite-checkpoints is not set. "
                    "Skipping saving."
                )
                dist.barrier()
                return
            else:
                print(f"WARNING: Overwriting existing checkpoint directory {checkpoint_dir}.")

        print(f"Saving checkpoint at iteration {iteration} to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    dist.barrier()

    # For FSDP v2, use the new state_dict API
    use_new_api = version.parse(torch.__version__) >= version.parse("2.4")

    if use_new_api:
        try:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

            # Get sharded state dicts for both model and optimizer
            model_state_dict, optimizer_state_dict = get_state_dict(
                model,
                optimizer,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=False,
                ),
            )

            state_dict = {
                "model": model_state_dict,
                "optim": optimizer_state_dict,
            }

            # Determine if we should use safetensors
            use_safetensors = getattr(args, "save_safe_serialization", False)

            if use_safetensors:
                try:
                    from torch.distributed.checkpoint import HuggingFaceStorageWriter

                    # Create FQN to index mapping for safetensors
                    fqn_to_index_mapping = {}
                    for key in state_dict["model"].keys():
                        fqn_to_index_mapping[f"model.{key}"] = 0

                    # Flatten optimizer state for safetensors
                    flattened_optim = _flatten_optimizer_state(optimizer_state_dict)
                    for key in flattened_optim.keys():
                        fqn_to_index_mapping[f"optim.{key}"] = 0

                    # Replace optimizer state dict with flattened version
                    state_dict["optim"] = flattened_optim

                    storage_writer = HuggingFaceStorageWriter(
                        path=checkpoint_dir,
                        fqn_to_index_mapping=fqn_to_index_mapping,
                    )

                    if dist.get_rank() == 0:
                        print("Saving with HuggingFace safetensors format")

                    dist_cp.save(
                        state_dict=state_dict,
                        storage_writer=storage_writer,
                    )

                except ImportError:
                    if dist.get_rank() == 0:
                        print("WARNING: HuggingFaceStorageWriter not available. Using standard format.")
                    storage_writer = dist_cp.FileSystemWriter(checkpoint_dir)
                    dist_cp.save(
                        state_dict=state_dict,
                        storage_writer=storage_writer,
                    )
            else:
                # Standard distributed checkpoint
                storage_writer = dist_cp.FileSystemWriter(checkpoint_dir)
                dist_cp.save(
                    state_dict=state_dict,
                    storage_writer=storage_writer,
                )

        except ImportError:
            if dist.get_rank() == 0:
                print("WARNING: New state_dict API not available, using fallback.")
            use_new_api = False

    if not use_new_api:
        # Fallback for FSDP v1
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
                "optim": FSDP.optim_state_dict(model, optimizer),
            }

            storage_writer = dist_cp.FileSystemWriter(checkpoint_dir)
            dist_cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=storage_writer,
            )

    # Save tokenizer and training state only on rank 0
    if dist.get_rank() == 0:
        tokenizer.save_pretrained(checkpoint_dir)

        training_state = {
            "iteration": iteration,
            "global_step": global_step,
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        print(f"Checkpoint saved to {checkpoint_dir}")

    dist.barrier()


def load_checkpoint(args, model, optimizer):
    """Load a checkpoint using FSDP distributed checkpointing.

    Compatible with both FSDP v1 and v2 (composable fully_shard).
    """
    loaded_rollout_id = -1
    global_step = 0
    checkpoint_path = args.load

    if not checkpoint_path:
        return loaded_rollout_id, global_step

    if dist.get_rank() == 0:
        print(f"Loading checkpoint from {checkpoint_path}")

    # Detect if checkpoint is in safetensors format
    is_safetensors = False
    if dist.get_rank() == 0:
        if os.path.exists(checkpoint_path):
            safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")]
            is_safetensors = len(safetensors_files) > 0
            if is_safetensors:
                print("Detected safetensors format checkpoint")

    # Broadcast format detection to all ranks
    is_safetensors_tensor = torch.tensor([1 if is_safetensors else 0], dtype=torch.long, device="cpu")
    dist.broadcast(is_safetensors_tensor, src=0)
    is_safetensors = bool(is_safetensors_tensor[0].item())

    # For FSDP v2, use the new state_dict API
    use_new_api = version.parse(torch.__version__) >= version.parse("2.4")

    if use_new_api:
        try:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

            # Get empty state dicts with the right structure
            model_state_dict, optimizer_state_dict = get_state_dict(
                model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=False)
            )

            # Load based on format
            if is_safetensors:
                try:
                    from torch.distributed.checkpoint import HuggingFaceStorageReader

                    # Load flattened optimizer state from safetensors
                    state_dict = {
                        "model": model_state_dict,
                        "optim": {},  # Will be populated with flattened keys
                    }

                    storage_reader = HuggingFaceStorageReader(path=checkpoint_path)
                    dist_cp.load(state_dict=state_dict, storage_reader=storage_reader)

                    # Unflatten the optimizer state dict
                    try:
                        optimizer_state_dict = _unflatten_optimizer_state(state_dict["optim"])
                    except (KeyError, IndexError):
                        # Fallback for older safetensors checkpoints without flattened optimizer state
                        if dist.get_rank() == 0:
                            print("WARNING: Failed to unflatten optimizer state. Assuming non-flattened format.")
                        optimizer_state_dict = state_dict["optim"]

                except ImportError:
                    if dist.get_rank() == 0:
                        print("WARNING: HuggingFaceStorageReader not available. Using standard reader.")

                    state_dict = {
                        "model": model_state_dict,
                        "optim": optimizer_state_dict,
                    }
                    storage_reader = dist_cp.FileSystemReader(checkpoint_path)
                    dist_cp.load(state_dict=state_dict, storage_reader=storage_reader)
            else:
                # Standard format - load directly
                state_dict = {
                    "model": model_state_dict,
                    "optim": optimizer_state_dict,
                }
                storage_reader = dist_cp.FileSystemReader(checkpoint_path)
                dist_cp.load(state_dict=state_dict, storage_reader=storage_reader)

            # Apply the loaded state
            set_state_dict(
                model,
                optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=optimizer_state_dict,
                options=StateDictOptions(full_state_dict=False, cpu_offload=False),
            )

            if dist.get_rank() == 0:
                print("Loaded model and optimizer state.")

        except ImportError:
            if dist.get_rank() == 0:
                print("WARNING: New state_dict API not available, using fallback.")
            use_new_api = False

    if not use_new_api:
        # Fallback for FSDP v1
        from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {"model": model.state_dict()}

            storage_reader = dist_cp.FileSystemReader(checkpoint_path)
            dist_cp.load_state_dict(state_dict=state_dict, storage_reader=storage_reader)
            model.load_state_dict(state_dict["model"])

            if dist.get_rank() == 0:
                print("Loaded model state.")

            # Load optimizer state
            try:
                optim_state = load_sharded_optimizer_state_dict(
                    model_state_dict=state_dict["model"],
                    optimizer_key="optim",
                    storage_reader=storage_reader,
                )

                flattened_osd = FSDP.optim_state_dict_to_load(model, optimizer, optim_state["optim"])
                optimizer.load_state_dict(flattened_osd)

                if dist.get_rank() == 0:
                    print("Loaded optimizer state.")
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"Optimizer state not found or failed to load: {e}, skipping.")

    # Load training state on rank 0 and broadcast
    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        if dist.get_rank() == 0:
            training_state = torch.load(training_state_path, map_location="cpu")
            loaded_rollout_id = training_state.get("iteration", -1)
            global_step = training_state.get("global_step", 0)
            print(f"Loaded training state: iteration={loaded_rollout_id}, global_step={global_step}")

        # Broadcast to all ranks with safe type handling
        training_state_tensor = torch.tensor([loaded_rollout_id, global_step], dtype=torch.int64, device="cpu")
        dist.broadcast(training_state_tensor, src=0)
        loaded_rollout_id = training_state_tensor[0].item()
        global_step = training_state_tensor[1].item()
    else:
        if dist.get_rank() == 0:
            print("Training state not found, skipping.")

    dist.barrier()
    return loaded_rollout_id, global_step


def _flatten_optimizer_state(optimizer_state_dict):
    """Flatten optimizer state dict for safetensors compatibility."""
    flattened = {}

    # Flatten state dictionary
    for param_name, param_states in optimizer_state_dict.get("state", {}).items():
        for state_key, state_value in param_states.items():
            fqn = f"state.{param_name}.{state_key}"
            flattened[fqn] = state_value

    # Flatten param_groups
    if "param_groups" in optimizer_state_dict:
        param_groups = optimizer_state_dict["param_groups"]
        for group_idx, group in enumerate(param_groups):
            for key, value in group.items():
                # Include all serializable values, not just tensors/ints/floats
                if isinstance(value, (torch.Tensor, int, float, bool, str)) or value is None:
                    fqn = f"param_groups.{group_idx}.{key}"
                    flattened[fqn] = value

    return flattened


def _unflatten_optimizer_state(flattened_dict):
    """Unflatten optimizer state dict from safetensors format."""
    state = {}
    param_groups = []

    for fqn, value in flattened_dict.items():
        parts = fqn.split(".")

        if parts[0] == "state":
            # Reconstruct state dictionary: state.<param_name>.<state_key>
            param_name = parts[1]
            state_key = parts[2]

            if param_name not in state:
                state[param_name] = {}
            state[param_name][state_key] = value

        elif parts[0] == "param_groups":
            # Reconstruct param_groups: param_groups.<group_idx>.<key>
            group_idx = int(parts[1])
            key = parts[2]

            # Ensure param_groups list is large enough
            while len(param_groups) <= group_idx:
                param_groups.append({})

            param_groups[group_idx][key] = value

    return {"state": state, "param_groups": param_groups}
