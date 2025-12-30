import math
import torch

from megatron.core.utils import get_te_version, is_te_min_version

# Check if Transformer Engine is installed
HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611
    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    pass

# Check if Transformer Engine has MXFP8Tensor class

try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE_MXFP8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # MXFP8Tensor not found
    HAVE_TE_MXFP8TENSOR = False

# Check if Transformer Engine has class for fp8 tensors.
HAVE_TE_FP8_TENSOR_CLASS = False
if HAVE_TE:
    from transformer_engine.pytorch.constants import TE_DType_To_Torch
    if is_te_min_version("2.0"):
        # In TE2.x, QuantizedTensor is the base class for all different type of fp8 tensors,
        # including fp8 tensor for delayed scaling, current scaling and mxfp8, etc.
        from transformer_engine.pytorch.tensor import QuantizedTensor as FP8_TENSOR_CLASS
    else:
        from transformer_engine.pytorch.float8_tensor import Float8Tensor as FP8_TENSOR_CLASS

    HAVE_TE_FP8_TENSOR_CLASS = True
else:
    HAVE_TE_FP8_TENSOR_CLASS = False
    FP8_TENSOR_CLASS = None

default_quantization_config = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [
      128,
      128
    ]
}

def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor.

    Note that in TE2.x, in order to support more recipes, the design of the fp8 tensor class has
    changed. Now Float8Tensor is only used for current scaling and delayed scaling. And mxfp8
    and blockwise scaling have their own fp8 tensor classes. These different fp8 tensor classes
    are both inherited from QuantizedTensor. So, for TE1.x, FP8_TENSOR_CLASS is Float8Tensor,
    and for TE2.x, FP8_TENSOR_CLASS is QuantizedTensor.
    """
    return HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS)

def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)

def get_fp8_weight_and_scale(tensor: torch.Tensor):
    fp8_metadata_dict = tensor.get_metadata()
    # te float8Tensor trick
    weight = fp8_metadata_dict['rowwise_data'].view(TE_DType_To_Torch[tensor._fp8_dtype])
    m, n = weight.shape

    # TE scale_shape will be round to multiple 4, need to reshape
    block_size = 128        # default block size is 128
    scale_m, scale_n = math.ceil(m / block_size), math.ceil(n / block_size)
    scale = fp8_metadata_dict['rowwise_scale_inv']

    return weight, scale[:scale_m, :scale_n].contiguous()
