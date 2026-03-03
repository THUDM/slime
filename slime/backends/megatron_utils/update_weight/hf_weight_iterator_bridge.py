import dataclasses


from slime.utils import megatron_bridge_utils
from slime.utils.misc import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


def _patch_bridge_expert_cache_to_cpu():
    """Monkey-patch GPTOSSBridge class to cache expert weights on CPU.

    This avoids GPU OOM when torch.cat merges all experts, especially in
    colocated mode where SGLang and Megatron share the same GPU.
    """
    try:
        from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
    except ImportError:
        return

    if getattr(GPTOSSBridge, "_cpu_cache_patched", False):
        return

    _orig = GPTOSSBridge.maybe_modify_converted_hf_weight

    def _patched(self, task, converted_weights_dict):
        cpu_dict = {k: v.cpu() for k, v in converted_weights_dict.items()}
        result = _orig(self, task, cpu_dict)
        # Move merged result back to GPU for CUDA IPC serialization
        return {k: v.cuda() for k, v in result.items()} if result else result

    GPTOSSBridge.maybe_modify_converted_hf_weight = _patched
    GPTOSSBridge._cpu_cache_patched = True


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        import slime_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        _patch_bridge_expert_cache_to_cpu()

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            def _streaming_quantized():
                for hf_param_name, weight, megatron_param_name in named_weights:
                    processed_weight = postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    )
                    converted_named_params = [(hf_param_name, processed_weight)]
                    quantized_batch = quantize_params(
                        args=self.args,
                        megatron_name=megatron_param_name,
                        converted_named_params=converted_named_params,
                        quantization_config=self.quantization_config,
                    )
                    yield from quantized_batch

            yield from chunk_named_params_by_size(
                _streaming_quantized(), chunk_size=self.args.update_weight_buffer_size
            )


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
