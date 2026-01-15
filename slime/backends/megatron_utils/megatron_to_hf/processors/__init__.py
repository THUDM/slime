from .padding_remover import remove_padding
from .quantizer_compressed_tensors import quantize_params_compressed_tensors
from .quantizer_fp8 import quantize_params_fp8

__all__ = ["remove_padding", "quantize_param", "quantize_params_fp8", "quantize_params_compressed_tensors"]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params
    elif quantization_config["quant_method"] == "fp8":
        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "compressed-tensors":
        # only int4 at the moment.
        return quantize_params_compressed_tensors(converted_named_params, quantization_config)
