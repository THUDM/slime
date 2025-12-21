import dataclasses
import re
import torch

from slime.utils import megatron_bridge_utils
from slime.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge
        import slime_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint)

    def merge_expert_weights_from_named_tuples(self, named_weights):
        # Pattern to match megatron param names ending with weight<NUMBER>
        expert_pattern = re.compile(r"^(.+\.weight)(\d+)$")

        # Group by megatron param base name
        expert_groups = {}  # base_name -> [(expert_idx, hf_param_name, weight, megatron_param_name)]
        non_expert_weights = []  # (hf_param_name, weight)

        for hf_param_name, weight, megatron_param_name in named_weights:
            match = expert_pattern.match(megatron_param_name)
            if match:
                base_name = match.group(1)  # e.g., "...weight"
                expert_idx = int(match.group(2))  # e.g., 0, 1, 2

                if base_name not in expert_groups:
                    expert_groups[base_name] = []

                expert_groups[base_name].append((expert_idx, hf_param_name, weight, megatron_param_name))
            else:
                # Not an expert weight, keep as is (only hf_param_name and weight)
                non_expert_weights.append((hf_param_name, weight))

        # Merge expert weights
        merged_weights = []
        for base_name, expert_list in expert_groups.items():
            # Sort by expert index
            expert_list.sort(key=lambda x: x[0])

            # Assert all experts map to the same hf_param_name
            hf_param_names = [item[1] for item in expert_list]
            assert all(name == hf_param_names[0] for name in hf_param_names), (
                f"Expert weights with base megatron name '{base_name}' map to different HF param names: "
                f"{set(hf_param_names)}. Megatron names: {[item[3] for item in expert_list]}"
            )

            # Stack tensors along first dimension
            tensors = [item[2] for item in expert_list]  # item[2] is the weight

            # Assert all tensors are on the same device and have the same dtype
            devices = [tensor.device for tensor in tensors]
            dtypes = [tensor.dtype for tensor in tensors]
            assert all(device == devices[0] for device in devices), (
                f"Expert weights with base megatron name '{base_name}' are on different devices: "
                f"{set(str(d) for d in devices)}"
            )
            assert all(dtype == dtypes[0] for dtype in dtypes), (
                f"Expert weights with base megatron name '{base_name}' have different dtypes: "
                f"{set(str(d) for d in dtypes)}"
            )

            # Move to CPU to avoid OOM during stacking, then move back to original device
            original_device = devices[0]
            original_dtype = dtypes[0]
            tensors_cpu = [tensor.cpu() for tensor in tensors]
            merged_tensor = torch.stack(tensors_cpu, dim=0)
            merged_tensor = merged_tensor.to(device=original_device, dtype=original_dtype)

            # Use the common hf_param_name
            merged_weights.append((hf_param_names[0], merged_tensor))

        # Combine non-expert and merged expert weights
        return non_expert_weights + merged_weights

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            named_weights = [
                (
                    hf_param_name,
                    postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    ),
                    megatron_param_name,
                )
                for hf_param_name, weight, megatron_param_name in named_weights
            ]

            named_weights = self.merge_expert_weights_from_named_tuples(named_weights)

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
