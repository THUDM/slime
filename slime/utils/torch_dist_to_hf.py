"""Load a Megatron torch_dist checkpoint and save as HuggingFace safetensors.

Shared by tools/convert_torch_dist_to_hf.py (CLI) and the --save-hf fallback
in slime.backends.megatron_utils.model (runtime). Kept free of global side
effects so it is safe to import during training.
"""

import json
import os
import pickle
import re
import shutil
import time

import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from typing_extensions import override

from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding


class _UnpicklerWrapper(pickle.Unpickler):
    """Deserialise checkpoint metadata without requiring Megatron/GLM classes."""

    @override
    def find_class(self, mod_name, name):
        class _Dummy:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return _Dummy
        return super().find_class(mod_name, name)


class StorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as f:
            metadata = _UnpicklerWrapper(f).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class LoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == num_experts
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            yield expert_name, param[expert_id]
    else:
        yield name, param


def get_layer_param(args, name, param):
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        assert param.shape[0] == num_layers
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            yield from get_expert_param(args, layer_name, param[layer_id])
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args, state_dict):
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)


def load_torch_dist(ckpt_dir: str) -> tuple:
    """Load a torch_dist checkpoint. Returns (megatron_args, state_dict)."""
    megatron_args = torch.load(os.path.join(ckpt_dir, "common.pt"), weights_only=False)["args"]
    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=StorageReader(ckpt_dir),
        planner=LoadPlanner(),
        no_dist=True,
    )
    return megatron_args, state_dict


def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None, origin_hf_dir=None):
    print(f"start saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    current_size = 0
    total_size = 0
    modeltensors = [{}]
    converted_names = set()
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        for converted_name, converted_param in convert_to_hf(args, model_name, name, param):
            converted_names.add(converted_name)
            tensor_size = converted_param.numel() * converted_param.element_size()
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param
            current_size += tensor_size
            total_size += tensor_size

    if origin_hf_dir is not None:
        safetensors_files = [f for f in os.listdir(origin_hf_dir) if f.endswith(".safetensors")]
        for filename in safetensors_files:
            with safetensors.safe_open(os.path.join(origin_hf_dir, filename), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k not in converted_names:
                        print(f"add {k} from origin hf checkpoint")
                        converted_param = f.get_tensor(k)
                        converted_names.add(k)
                        tensor_size = converted_param.numel() * converted_param.element_size()
                        if tensor_size + current_size > chunk_size:
                            modeltensors.append({})
                            current_size = 0
                        modeltensors[-1][k] = converted_param
                        current_size += tensor_size
                    total_size += tensor_size

    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}
    num_files = len(modeltensors)
    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        for key in tensors:
            metadata["weight_map"][key] = filename
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"{index_filepath} saved.")

    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        t = time.time()
        safetensors.torch.save_file(tensors, os.path.join(output_dir, filename))
        print(f"{filename} saved in {time.time() - t:.2f} sec.")


def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        src = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(output_dir, filename)
        shutil.copy(src, dst)
