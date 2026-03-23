import argparse
import json
import multiprocessing
import os
import pickle
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from tqdm import tqdm
from transformers import AutoConfig
from typing_extensions import override

from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding


class DummyClass:
    def __init__(self, *args, **kwargs):
        pass


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    def __init__(self, path, load_id=None, max_workers=None):
        super().__init__(path, load_id)
        self.max_workers = max_workers
        self._read_cache = {} if max_workers and max_workers > 1 else None
        self._cache_lock = threading.Lock() if max_workers and max_workers > 1 else None

    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata

    def _collect_read_operations(self, plan):
        read_ops = []

        def collect_from_storage_metadata(storage_item, base_path=""):
            if isinstance(storage_item, dist_cp.metadata.BytesStorageMetadata):
                full_path = self.fs.concat_path(self.path, storage_item.relative_path)
                read_ops.append(("bytes", storage_item.relative_path, full_path, None, None))
            elif isinstance(storage_item, dist_cp.metadata.ChunkStorageMetadata):
                full_path = self.fs.concat_path(self.path, storage_item.relative_path)
                read_ops.append(
                    ("chunk", storage_item.relative_path, full_path, storage_item.offset, storage_item.length)
                )
            elif isinstance(storage_item, dist_cp.metadata.TensorStorageMetadata):
                if hasattr(storage_item, "chunks"):
                    for chunk in storage_item.chunks:
                        collect_from_storage_metadata(chunk)

        if hasattr(plan, "items"):
            for item in plan.items:
                collect_from_storage_metadata(item)
        elif hasattr(plan, "state_dict_metadata"):
            for _key, metadata in plan.state_dict_metadata.items():
                if hasattr(metadata, "storage"):
                    collect_from_storage_metadata(metadata.storage)

        return read_ops

    def _parallel_preload_files(self, metadata=None, keys=None, state_dict_metadata_dict=None):
        if not self.max_workers or self.max_workers <= 1:
            return None

        keys_set = set(keys) if keys else None

        read_ops = []

        if state_dict_metadata_dict is not None:
            metadata_dict = state_dict_metadata_dict
        elif metadata is not None and hasattr(metadata, "state_dict_metadata"):
            metadata_dict = metadata.state_dict_metadata
        else:
            if metadata is None:
                metadata = self.read_metadata()
            metadata_dict = metadata.state_dict_metadata if hasattr(metadata, "state_dict_metadata") else {}

        for key, tensor_meta in metadata_dict.items():
            if "optimizer" in key or "_state" in key:
                continue
            if keys_set is not None and key not in keys_set:
                continue

            if hasattr(tensor_meta, "storage"):
                storage = tensor_meta.storage
                if isinstance(storage, dist_cp.metadata.BytesStorageMetadata):
                    full_path = self.fs.concat_path(self.path, storage.relative_path)
                    read_ops.append(("bytes", storage.relative_path, full_path, None, None))
                elif isinstance(storage, dist_cp.metadata.ChunkStorageMetadata):
                    full_path = self.fs.concat_path(self.path, storage.relative_path)
                    read_ops.append(("chunk", storage.relative_path, full_path, storage.offset, storage.length))
                elif hasattr(storage, "chunks"):
                    for chunk in storage.chunks:
                        if isinstance(chunk, dist_cp.metadata.ChunkStorageMetadata):
                            full_path = self.fs.concat_path(self.path, chunk.relative_path)
                            read_ops.append(("chunk", chunk.relative_path, full_path, chunk.offset, chunk.length))

        if not read_ops:
            return None

        # Group by file path - for chunked files, we'll cache the whole file
        file_ops = {}
        for op_type, rel_path, _full_path, offset, length in read_ops:
            if rel_path not in file_ops:
                file_ops[rel_path] = []
            file_ops[rel_path].append((op_type, offset, length))

        # Parallel read files - for files with chunks, read the whole file
        def read_file(rel_path, full_path, ops):
            try:
                with self.fs.create_stream(full_path, "rb") as stream:
                    has_chunks = any(op[0] == "chunk" and op[1] is not None for op in ops)
                    if has_chunks:
                        data = stream.read()
                        return rel_path, data, True  # True indicates full file
                    else:
                        data = stream.read()
                        return rel_path, data, False
            except Exception as e:
                print(f"Error reading {rel_path}: {e}")
                return None

        print(f"Parallel pre-loading {len(file_ops)} files with {self.max_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(read_file, rel_path, full_path, ops): rel_path
                for rel_path, (full_path, ops) in [
                    (r, (self.fs.concat_path(self.path, r), ops)) for r, ops in file_ops.items()
                ]
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-loading files"):
                result = future.result()
                if result:
                    rel_path, data, is_full_file = result
                    with self._cache_lock:
                        # Cache by file path - for chunked files, store the full file
                        self._read_cache[rel_path] = data

        return self._read_cache

    @override
    def read_data(self, plan, planner):
        """Override read_data to use parallel reading when enabled"""
        if self.max_workers and self.max_workers > 1 and self._read_cache is not None and len(self._read_cache) > 0:
            original_create_stream = self.fs.create_stream

            def cached_create_stream(path, mode="rb"):
                if mode == "rb":
                    try:
                        if path.startswith(self.path):
                            rel_path = os.path.relpath(path, self.path)
                        else:
                            filename = os.path.basename(path)
                            rel_path = None
                            for cached_path in self._read_cache.keys():
                                if os.path.basename(cached_path) == filename:
                                    rel_path = cached_path
                                    break
                            if rel_path is None:
                                return original_create_stream(path, mode)

                        if rel_path in self._read_cache:
                            from io import BytesIO

                            return BytesIO(self._read_cache[rel_path])
                    except (ValueError, KeyError, OSError):
                        pass
                return original_create_stream(path, mode)

            self.fs.create_stream = cached_create_stream
            try:
                result = super().read_data(plan, planner)
            finally:
                self.fs.create_stream = original_create_stream
            return result

        return super().read_data(plan, planner)


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    def __init__(self, keys=None):
        super().__init__()
        self.keys = set(keys) if keys else None

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
            if self.keys is not None and k not in self.keys:
                continue
            print(f"find {k} in torch_dist ckpt")
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
            expert_param = param[expert_id]
            yield expert_name, expert_param
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
            layer_param = param[layer_id]
            yield from get_expert_param(args, layer_name, layer_param)
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args, state_dict):
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)


def process_param(args, model_name, name, param, vocab_size=None):
    if vocab_size:
        param = remove_padding(name, param, vocab_size)
    converted_named_tensors = list(convert_to_hf(args, model_name, name, param))
    return converted_named_tensors


def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None, max_workers=1, worker_id=None):
    # for slime update_weight compatible
    args.sglang_enable_ep_moe = False

    print(f"start saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    param_list = list(get_named_params(args, state_dict))
    print(f"Total parameters to process: {len(param_list)}")

    all_converted_tensors = []
    lock = threading.Lock()

    def process_and_collect(name_param_pair):
        name, param = name_param_pair
        try:
            converted = process_param(args, model_name, name, param, vocab_size)
            return converted
        except Exception as e:
            print(f"Error processing {name}: {e}")
            return []

    print(f"Processing with {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_collect, (name, param)): name for name, param in param_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting parameters"):
            converted = future.result()
            with lock:
                all_converted_tensors.extend(converted)

    current_size = 0
    total_size = 0
    modeltensors = [{}]
    for converted_name, converted_param in all_converted_tensors:
        tensor_size = converted_param.numel() * converted_param.element_size()
        if tensor_size + current_size > chunk_size:
            modeltensors.append({})
            current_size = 0
        modeltensors[-1][converted_name] = converted_param
        current_size += tensor_size
        total_size += tensor_size

    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}

    num_files = len(modeltensors)
    file_prefix = f"worker_{worker_id}_" if worker_id is not None else ""
    index_prefix = f"worker_{worker_id}_" if worker_id is not None else ""

    for i, tensors in enumerate(modeltensors):
        filename = f"{file_prefix}model-{i:05d}-of-{num_files:05d}.safetensors"
        for key in tensors.keys():
            metadata["weight_map"][key] = filename
    index_filepath = os.path.join(output_dir, f"{index_prefix}model.safetensors.index.json")
    json.dump(metadata, open(index_filepath, "w"), indent=2)
    print(f"{index_filepath} saved.")

    def save_file(i, tensors):
        filename = f"{file_prefix}model-{i:05d}-of-{num_files:05d}.safetensors"
        t = time.time()
        filepath = os.path.join(output_dir, filename)
        safetensors.torch.save_file(tensors, filepath)
        return filename, time.time() - t

    with ThreadPoolExecutor(max_workers=min(max_workers, num_files)) as executor:
        futures = [executor.submit(save_file, i, tensors) for i, tensors in enumerate(modeltensors)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving files"):
            filename, elapsed = future.result()
            print(f"{filename} saved in {elapsed:.2f} sec.")


def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)


def conversion_worker(
    worker_id,
    keys,
    args,
    megatron_args,
    model_name,
    state_dict_metadata_dict=None,
    load_max_workers=2,
    save_max_workers=16,
):
    print(f"Worker {worker_id} starting with {len(keys)} keys")
    planner = EmptyStateDictLoadPlanner(keys)
    state_dict = {}

    storage_reader = WrappedStorageReader(args.input_dir, max_workers=load_max_workers)

    if load_max_workers and load_max_workers > 1:
        print(f"Worker {worker_id}: Pre-loading files with {load_max_workers} workers...")
        storage_reader._parallel_preload_files(keys=keys, state_dict_metadata_dict=state_dict_metadata_dict)

    print(f"Worker {worker_id}: Loading state dict...")
    t = time.time()
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=storage_reader,
        planner=planner,
        no_dist=True,
    )
    print(f"Worker {worker_id}: State dict loaded in {time.time()-t:.2f} sec.")

    save_tensors(
        megatron_args,
        model_name,
        state_dict,
        args.output_dir,
        args.chunk_size,
        args.vocab_size,
        max_workers=save_max_workers,
        worker_id=worker_id,
    )

    index_file = os.path.join(args.output_dir, f"worker_{worker_id}_model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            idx = json.load(f)
        return idx
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        default=None,
        help="use the origin hf dir to copy files like tokenizer, config.json, etc.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5 * 1024**3,
        help="Chunk size for saving tensors, default is 2GB.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocab size for removing padding, if applicable. If not provided, no padding will be removed.",
    )
    parser.add_argument(
        "--load-max-workers",
        type=int,
        default=2,
        help="Number of worker threads for parallel file loading. Default is 2.",
    )
    parser.add_argument(
        "--save-max-workers",
        type=int,
        default=16,
        help="Number of worker threads for parallel tensor conversion and saving. Default is 16.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError(
            "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
        )

    if args.model_name is None:
        hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
        args.model_name = type(hf_config).__name__.lower()

    megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]

    load_max_workers = args.load_max_workers
    save_max_workers = args.save_max_workers

    reader = WrappedStorageReader(args.input_dir)
    metadata = reader.read_metadata()
    all_keys = [k for k in metadata.state_dict_metadata.keys() if "optimizer" not in k and "_state" not in k]

    # Group paired keys (linear_q_down_proj and linear_kv_down_proj) together
    # These will be converted to q_a_proj and kv_a_proj_with_mqa later
    paired_keys = set()
    key_groups = []  # Each element is either a single key or a pair of keys

    for key in all_keys:
        if key in paired_keys:
            continue

        # Check if this is a linear_q_down_proj or linear_kv_down_proj key
        if "linear_q_down_proj" in key or "linear_kv_down_proj" in key:
            # Find its pair
            if "linear_q_down_proj" in key:
                pair_key = key.replace("linear_q_down_proj", "linear_kv_down_proj")
            else:
                pair_key = key.replace("linear_kv_down_proj", "linear_q_down_proj")

            # If pair exists in all_keys, group them together
            if pair_key in all_keys:
                key_groups.append([key, pair_key])
                paired_keys.add(key)
                paired_keys.add(pair_key)
            else:
                key_groups.append([key])
        else:
            key_groups.append([key])

    # Distribute key groups to workers (round-robin to balance load)
    num_workers = load_max_workers
    key_chunks = [[] for _ in range(num_workers)]

    for i, key_group in enumerate(key_groups):
        worker_idx = i % num_workers
        key_chunks[worker_idx].extend(key_group)

    # Remove empty chunks
    key_chunks = [chunk for chunk in key_chunks if chunk]
    num_workers = len(key_chunks)

    print(
        f"Total keys: {len(all_keys)}, Paired groups: {len([g for g in key_groups if len(g) > 1])}, Workers: {num_workers}"
    )

    state_dict_metadata_dict = metadata.state_dict_metadata if hasattr(metadata, "state_dict_metadata") else {}

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        results = pool.starmap(
            conversion_worker,
            [
                (
                    i,
                    key_chunks[i],
                    args,
                    megatron_args,
                    args.model_name,
                    state_dict_metadata_dict,
                    load_max_workers,
                    save_max_workers,
                )
                for i in range(num_workers)
            ],
        )

    final_weight_map = {}
    total_size = 0
    final_file_index = 1

    for i, res in enumerate(results):
        if not res:
            continue

        w_map = res["weight_map"]
        total_size += res["metadata"]["total_size"]

        worker_files = sorted(list(set(w_map.values())))
        file_rename_map = {}
        for wf in worker_files:
            new_name = f"model-{final_file_index:05d}.safetensors"
            final_file_index += 1
            file_rename_map[wf] = new_name

            src = os.path.join(args.output_dir, wf)
            dst = os.path.join(args.output_dir, new_name)
            if os.path.exists(src):
                shutil.move(src, dst)

        for param_name, old_fname in w_map.items():
            final_weight_map[param_name] = file_rename_map[old_fname]

        temp_index = os.path.join(args.output_dir, f"worker_{i}_model.safetensors.index.json")
        if os.path.exists(temp_index):
            os.remove(temp_index)

    total_files = final_file_index - 1
    final_weight_map_fixed = {}
    for i in range(1, total_files + 1):
        old_name = f"model-{i:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{total_files:05d}.safetensors"
        old_path = os.path.join(args.output_dir, old_name)
        new_path = os.path.join(args.output_dir, new_name)
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
        for k, v in final_weight_map.items():
            if v == old_name:
                final_weight_map_fixed[k] = new_name

    index_data = {"metadata": {"total_size": total_size}, "weight_map": final_weight_map_fixed}
    json.dump(index_data, open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w"), indent=2)
    print("Model converted and saved.")

    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)
