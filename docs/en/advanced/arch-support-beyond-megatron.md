# Supporting Model Architectures Beyond Megatron-LM

While the Megatron-LM framework is highly efficient for parallel training, it can lack the flexibility to support rapidly evolving model architectures like Qwen3Next. Natively supporting the unique structures of these models, such as Gated-Delta-Net, often requires invasive and time-consuming modifications to Megatron's core codebase.

To accelerate the adoption of these cutting-edge models, slime introduces a more agile approach: **instead of deeply re-engineering Megatron, we directly import and wrap the model's official HuggingFace implementation**, embedding it as a "black-box" module into Megatron's parallel training pipeline.

This document uses Qwen3Next 80B-A3B as an example to illustrate this concept.

## Principle and Core Components

Megatron's model instantiation is a two-step process: first, it generates a "layer specification" (`ModuleSpec`) based on the configuration, and then it instantiates the actual PyTorch modules according to that spec.

slime leverages this mechanism by **hijacking the spec generation stage to replace Megatron's native modules** with an external implementation (in this case, from HuggingFace). This process involves the coordination of three core components:

1.  **Replacing the Megatron Module Spec**
    This is the entry point for our solution. We use a custom function (e.g., `get_qwen3_next_spec`) to modify the standard `ModuleSpec`, swapping out Megatron's native Attention layer with our custom wrapper.
    * **Implementation**: It retrieves the standard Decoder Block Spec, points its `self_attention` field to our custom module, and enables model-specific configurations like `qk_layernorm` as needed.
    * **Corresponding File**: `slime_plugins/models/qwen3_next.py`

2.  **Wrapping the HuggingFace Implementation**
    The spec modified in the previous step now points to a wrapper layer, such as `HuggingfaceAttention`. This layer inherits from Megatron's `MegatronModule`. Its core responsibility is to act as a bridge, handling the data alignment required by parallelism strategies (like sequence parallelism), and then internally calling the native `Qwen3NextAttention` module loaded from HuggingFace.
    * **Corresponding File**: `slime_plugins/models/hf_attention.py`

3.  **Aligning Model Weights**
    Once the model architecture is integrated, we must ensure that the weights can be loaded correctly. We use the [mbridge](https://github.com/ISEEKYAN/mbridge) library, through our `Qwen3NextBridge`, to establish a naming map between the HuggingFace checkpoint and Megatron's parameters, enabling seamless, bidirectional conversion.
    * **Corresponding File**: `slime_plugins/mbridge/qwen3_next.py`

Through the coordination of these three components, we can successfully run a complex model architecture not natively supported by Megatron—using its HuggingFace implementation as the vehicle—on top of Megatron's parallel framework. This is achieved while fully retaining all key capabilities like model parallelism, MoE acceleration, and pipeline scheduling.

## Current Limitations

* This approach does not currently support Tensor Parallelism (TP) within the replaced module itself (e.g., the Attention layer in this case).
* **Impact**: In most large-scale MoE models, the parameter count of the Attention layer is relatively small, so this limitation typically has a minimal effect on memory footprint and training throughput.
* **Alternative**: If TP for the module is critical, the only alternative is to revert to the more invasive approach of modifying Megatron's native implementation.
