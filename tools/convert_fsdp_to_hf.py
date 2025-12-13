import argparse
import os
import pickle
import shutil
import time

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig, AutoModelForCausalLM
from typing_extensions import override


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class WrappedStorageReader(dist_cp.FileSystemReader):
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


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
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
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _detect_model_dir(input_dir: str) -> str:
    model_dir = os.path.join(input_dir, "model")
    return model_dir if os.path.isdir(model_dir) else input_dir


def _load_fsdp_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _convert_fsdp_to_hf(origin_hf_dir: str, input_dir: str, output_dir: str) -> None:
    print(f"loading FSDP model from {input_dir}")
    t = time.time()
    state_dict = _load_fsdp_state_dict(input_dir)
    print(f"FSDP model loaded in {time.time()-t:.2f} sec.")

    model_state = {
        k.replace("model_state.model.", "", 1).replace("model.", "", 1): v
        for k, v in state_dict.items()
        if isinstance(v, torch.Tensor) and (k.startswith("model_state.model.") or k.startswith("model."))
    }

    if not model_state:
        raise ValueError(
            "No model weights found in checkpoint. "
            "Please pass the checkpoint directory (e.g. iter_xxx or iter_xxx/model)."
        )

    config = AutoConfig.from_pretrained(origin_hf_dir, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_config(config)
    missing, unexpected = hf_model.load_state_dict(model_state, strict=False)
    print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Model weights saved to {output_dir}")


def copy_assets(origin_hf_dir: str, output_dir: str) -> None:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="The original Hugging Face model directory to load config/tokenizer assets.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    model_dir = _detect_model_dir(args.input_dir)
    _convert_fsdp_to_hf(args.origin_hf_dir, model_dir, args.output_dir)
    copy_assets(args.origin_hf_dir, args.output_dir)
