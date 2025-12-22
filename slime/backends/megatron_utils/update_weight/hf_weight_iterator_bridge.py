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

        import slime_plugins.megatron_bridge  # noqa: F401

        self._bridge = args.bridge

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            named_weights = (
                (
                    hf_param_name,
                    postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    ),
                )
                for hf_param_name, weight, megatron_param_name in named_weights
            )

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)


def _has_expert_weights(tasks):
    """Check if any task has param_name ending with weight<NUMBER>."""
    expert_pattern = re.compile(r"^.+\.weight\d+$")
    return any(task.param_weight is not None and expert_pattern.match(task.param_name) for task in tasks)


def _merge_expert_tasks(tasks, new_weight_dict):
    """Merge expert tasks (weight0, weight1, ...) into single tasks with stacked tensors."""
    expert_pattern = re.compile(r"^(.+\.weight)(\d+)$")
    expert_groups = {}  # (vp_stage, base_param_name) -> [(expert_idx, task)]
    non_expert_tasks = []

    for task in tasks:
        if task.param_weight is None:
            non_expert_tasks.append(task)
            continue

        match = expert_pattern.match(task.param_name)
        if match:
            base_param_name = match.group(1)
            expert_idx = int(match.group(2))
            group_key = (task.vp_stage, base_param_name)
            expert_groups.setdefault(group_key, []).append((expert_idx, task))
        else:
            non_expert_tasks.append(task)

    # Create merged tasks
    merged_tasks = []
    for (_, base_param_name), expert_list in expert_groups.items():
        expert_list.sort(key=lambda x: x[0])  # Sort by expert_idx numerically

        cpu_tensors = [
            new_weight_dict[f"vp_stages.{task.vp_stage}.{task.param_name}"].transpose(0, 1) for _, task in expert_list
        ]
        merged_tensor = torch.stack(cpu_tensors, dim=1).cuda()
        # Use first expert's task as template for merged task
        template_task = expert_list[0][1]  # expert_list = [(expert_idx, task), ...]
        merged_task = dataclasses.replace(template_task, param_name=base_param_name, param_weight=merged_tensor)
        merged_tasks.append(merged_task)

    return non_expert_tasks, merged_tasks


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    """Process conversion tasks, merging expert weights if present to avoid OOM."""
    all_tasks = list(vanilla_conversion_tasks)

    # Check if expert merging is needed
    if _has_expert_weights(all_tasks):
        non_expert_tasks, merged_tasks = _merge_expert_tasks(all_tasks, new_weight_dict)
    else:
        non_expert_tasks = all_tasks
        merged_tasks = []

    # Move weights to CUDA for non-expert tasks
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

    processed_non_expert = [_handle_one(task) for task in non_expert_tasks]
    return _MapWithLen(lambda x: x, processed_non_expert + merged_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
