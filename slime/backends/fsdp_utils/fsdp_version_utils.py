"""
FSDP v2 Utilities

This module provides utilities for FSDP v2 (fully_shard) support only.
FSDP v1 is no longer supported and will raise errors if detected.
"""

import torch
import logging
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


def verify_fsdp_v2_support():
    """
    Verify that FSDP v2 is available and import the necessary components.
    
    Returns:
        tuple: (fully_shard, FSDPModule) - The FSDP v2 components
        
    Raises:
        RuntimeError: If FSDP v2 is not available or FSDP v1 is detected
    """
    from packaging import version
    
    # Check PyTorch version compatibility
    if version.parse(torch.__version__) < version.parse("2.4"):
        raise RuntimeError(
            f"FSDP v2 requires PyTorch >= 2.4. Current version: {torch.__version__}. "
            "Please upgrade PyTorch to use FSDP v2."
        )
    
    # Import FSDP v2 components based on PyTorch version
    try:
        if version.parse(torch.__version__) >= version.parse("2.6"):
            from torch.distributed.fsdp import fully_shard, FSDPModule
        elif version.parse(torch.__version__) >= version.parse("2.4"):
            from torch.distributed._composable.fsdp import fully_shard, FSDPModule
        else:
            raise ImportError("FSDP v2 not available")
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import FSDP v2 components: {e}. "
            f"PyTorch version: {torch.__version__}. "
            "Please ensure FSDP v2 is properly installed."
        )
    
    # Check for deprecated FSDP v1 usage
    try:
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP_v1
        logger.warning(
            "FSDP v1 (FullyShardedDataParallel) is available but deprecated. "
            "This utility only supports FSDP v2 (fully_shard)."
        )
    except ImportError:
        pass  # FSDP v1 not available, which is fine
    
    logger.info(f"FSDP v2 verified successfully with PyTorch {torch.__version__}")
    return fully_shard, FSDPModule


def verify_model_is_fsdp_v2(model):
    """
    Verify that a model is wrapped with FSDP v2.
    
    Args:
        model: The model to verify
        
    Raises:
        RuntimeError: If model is not FSDP v2 or is FSDP v1
    """
    _, FSDPModule = verify_fsdp_v2_support()
    
    # Check for FSDP v1 (deprecated)
    try:
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP_v1
        if isinstance(model, FSDP_v1):
            raise RuntimeError(
                "FSDP v1 (FullyShardedDataParallel) is no longer supported. "
                "Please upgrade to FSDP v2 using fully_shard(). "
                f"Current PyTorch version: {torch.__version__}"
            )
    except ImportError:
        pass  # FSDP v1 not available, which is fine
    
    # Check for FSDP v2
    if not isinstance(model, FSDPModule):
        raise RuntimeError(
            "Model is not wrapped with FSDP v2 (fully_shard). "
            f"Model type: {type(model)}. "
            f"PyTorch version: {torch.__version__}. "
            "Please use fully_shard() to wrap your model."
        )
    
    logger.info("Model verified as FSDP v2")


def create_fsdp_v2_model(model):
    """Create an FSDP v2 model using fully_shard."""
    fully_shard, _ = verify_fsdp_v2_support()
    
    logger.info(f"Creating FSDP v2 model with PyTorch {torch.__version__}")
    # FSDP v2 using fully_shard (produces DTensor objects)
    return fully_shard(model)


def preprocess_tensor_for_update_weights(tensor):
    """
    Preprocess tensor for weight updates - FSDP v2 only (DTensor support).
    
    Args:
        tensor: The tensor to preprocess (DTensor or regular Tensor)
        
    Returns:
        torch.Tensor: Regular tensor ready for weight updates
    """
    if isinstance(tensor, DTensor):
        # FSDP v2 case - convert DTensor to full tensor
        return tensor.full_tensor()
    
    # Regular tensor - return as is
    return tensor
