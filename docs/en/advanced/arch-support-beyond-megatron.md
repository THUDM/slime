# Supporting Model Architectures Beyond Megatron-LM

slime can extend a Megatron model by replacing selected layer specifications (`ModuleSpec`) with a custom implementation. Qwen3-Next 80B-A3B uses this approach for its Gated DeltaNet (GDN) layers, while retaining Megatron's full-attention layers, MoE blocks, and pipeline scheduling.

## Principle and Core Components

The Qwen3-Next integration has three parts:

1. **Select the layers to replace.** `get_qwen3_next_spec` starts from Megatron's decoder block specification and replaces only the layers marked `linear_attention` in the Hugging Face configuration. It accounts for the layer offset of each pipeline stage. See [qwen3_next.py](https://github.com/THUDM/slime/blob/main/slime_plugins/models/qwen3_next.py).
2. **Run GDN within Megatron.** The GDN implementation adapts the Hugging Face model layout and uses FLA or the optional FlashQLA backend for the recurrent attention kernel. `HuggingfaceAttention` gathers sequence and context partitions before the replicated computation, then returns the local output partition. See [hf_attention.py](https://github.com/THUDM/slime/blob/main/slime_plugins/models/hf_attention.py).
3. **Load the model weights.** The checkpoint loader maps Hugging Face parameter names and tensor layouts to the custom modules and the retained Megatron layers. See [the Qwen3-Next weight loader](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/hf_to_megatron/qwen3_next.py).

The [Qwen3-Next example](../examples/qwen3-next-80B-A3B.md) contains the launch configuration and backend options.

## Current Limitations

The custom GDN module supports jobs using TP and CP, but its parameters and gathered computation are replicated across those ranks. Its projections are not sharded by TP, so increasing TP does not reduce that module's parameter memory or computation in the same way as a native tensor-parallel layer.

The current spec supports pipeline stage offsets, but rejects a custom `pipeline_model_parallel_layout`. A native parallel GDN implementation would be needed to remove the replication cost.
