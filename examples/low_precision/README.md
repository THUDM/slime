## FP8 training examples

This is an example of FP8 training and FP8 inference. Under FP8 training and inference, it can achieve more efficient inference throughput and lower training-inference mismatch, resulting in more stable training.

### Files

* `run-qwen3-4b-fp8.sh`: example launch script with Qwen3‑4B in FP8.

* `run-qwen3-30b-a3b-fp8-two-nodes.sh`: example launch script for running Qwen3‑30B‑A3B in FP8 across two nodes.

### Quick Start

1. Check if your training script is properly configured. 

For training tasks, we need to add these flags:
```bash
--fp8-format e4m3
--fp8-recipe blockwise
# --fp8-param-gather # [optional] Currently incompatible with CPU Adam
```
Then ensure the `NVTE_FP8_BLOCK_SCALING_FP32_SCALES` environment variable is enabled.

Note that only `Linear` and `GroupLinear` layers in TransformerEngine use fp8 format. `embedding` and `lm_head` remain in their original precision. If `--fp8-param-gather` is not enabled, weights in TransformerEngine remain in bf16 format, only being cast to fp8 format during `GEMM` or `GroupGEMM` operations.

2. Convert your HuggingFace model weights to FP8 format. 

You can use `tools/convert_hf_to_fp8.py` to convert bf16 weights to fp8 format. Ensure that the `--hf-checkpoint` parameter points to a directory where the `config.json` contains the correct `quantization_config`. slime will automatically use FP8 quantization during weight updates. 

3. Start FP8 training.

```
cd slime

# Qwen3‑4B FP8 training (single node)
bash examples/low_precision/run-qwen3-4b-fp8.sh

# Qwen3‑30B‑A3B FP8 training (two nodes)
bash examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh
```
Following the above command will launch FP8 training. 

4. Use the saved checkpoint for evaluation. 

Note that TransformerEngine does not specifically save FP8 quantized weights; the saved torch dist remains in original precision (usually bf16). If you want to evaluate under FP8, you need to convert the checkpoint from `torch_dist` to HuggingFace format, then convert to FP8 HuggingFace format.


### Quick Explanation

Here's a quick explanation of how FP8 training is currently implemented in slime:

1. Initialization: If FP8 recipe is enabled, layers will be built in FP8 context.

2. Training: During training, weights and activations are quantized online to nvfp8 format, and cuBLAS FP8 GEMM is called for various GEMM computations in forward and backward passes.

3. Weight updates: During RL weight updates, Megatron first dequantizes FP8 weights to bf16 format, then slime quantizes these bf16 weights to fp8 format and sends them to sglang. (This additional dequantization and quantization is not elegant, but we haven't modified the interface yet for framework compatibility.)

4. Save checkpoint: Similar to weight updates, if checkpoints need to be saved from the training engine, they will also be dequantized back to bf16 and saved to `torch_dist` format checkpoints.


### TODO

Currently, FP8 is far from being a complete feature and still has the following bugs, for examples:

- FP8 weights (`--fp8-param-gather`) can provide memory savings benefits, but currently FP8 weights must be used with TransformerEngine's FusedAdam, which conflicts with the commonly used Adam CPU offload technique in Megatron-LM.

The slime team will continue to collaborate with the NVIDIA team to contribute more complete FP8 training infrastructure to the community.


Here is a polished and professional version of your documentation.

I have corrected grammatical errors, improved the flow, standardizes the terminology (e.g., capitalizing "STE"), and clarified the instructions.

***

## INT4 Training Examples

This guide provides examples for INT4 STE (Straight-Through Estimator) training and INT4 inference. Utilizing INT4 inference significantly improves throughput, thereby accelerating the training pipeline (specifically during the rollout generation phase).

### Files

*   `run-moonlight-16B-A3B-int4.sh`: Launch script for **Moonlight-16B-A3B** (INT4) on 4x H200 GPUs.
*   `run-qwen3‑30B‑A3B-int4.sh`: Launch script for **Qwen3‑30B‑A3B** (INT4) on 8x H200 GPUs.
*   `run-qwen3-235B-A22B-int4.sh`: Launch script for **Qwen3-235B-A22B** (INT4) on 64x H200 GPUs.
*   `run-kimi-k2-Thinking-int4.sh`: Launch script for **Kimi-k2-Thinking** (INT4) on 256x H200 GPUs.

### Quick Start

#### 1. Configure Training Arguments
Ensure your training script is properly configured. For training tasks, you must add the following flag to your launch arguments:

```bash
--int4-params-rollout
```

#### 2. Convert HuggingFace Weights to INT4
First, download the PTQ (Post-Training Quantization) calibration dataset from HuggingFace:
[https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1)

Next, use the `tools/convert_hf_to_hf_int4.py` script to convert BF16 weights to INT4 format. Ensure that the `--hf-checkpoint` parameter points to a directory where `config.json` contains the correct `quantization_config`. Slime will automatically utilize INT4 quantization during weight updates.

```bash
python tools/convert_hf_to_hf_int4.py \
  --model_id /path/to/your/original/models \
  --output_dir /path/to/your/save/models \
  --local_data_path /path/to/your/wikitext
```

#### 3. Modifying the Transformer Engine Code

Maintaining a custom patch for the Transformer Engine (TE) introduces significant complexity to the project. Therefore, we strongly recommend using our pre-built Docker image. 
Alternatively, if you must use newest docker and code, follow the steps below to manually modify the TE code.

**Option A: Use the Pre-built Docker Image (Recommended)**

-   Use the provided Docker image which already contains the necessary environment configuration.

**Option B: Manual Modification**
- **Target File:**
        `/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/grouped_linear.py`
- **Instructions:**
        Add the helper functions and modify the `forward` method within the `_GroupedLinear` class as shown below:

```python
# 1. Add these helper functions (e.g., at the top of the file)
import os
import torch

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

def fake_int4_quantization_ste(x, block_size):
    m, n = x.shape
    block_size_m, block_size_n = block_size[0], block_size[1]

    m_padded = ceil_div(m, block_size_m) * block_size_m
    n_padded = ceil_div(n, block_size_n) * block_size_n

    x_padded = torch.zeros(
        (m_padded, n_padded),
        dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x

    x_view = x_padded.view(
        m_padded // block_size_m,
        block_size_m,
        n_padded // block_size_n,
        block_size_n
    )

    x_max = x_view.abs().float().amax(dim=(1, 3), keepdim=True)
    q_max = 7
    x_scale = x_max / q_max

    x_scale = x_scale.clamp(min=1e-5)

    x_div = x_view / x_scale
    x_round = torch.round(x_div)

    x_q_clamped = x_round.clamp(-q_max, q_max)

    x_dequant_view = x_q_clamped * x_scale

    x_dequant_full = x_dequant_view.view_as(x_padded)
    x_out = x_dequant_full[:m, :n].contiguous().to(x.dtype)

    return x + (x_out - x).detach()

# 2. Modify the forward method in the _GroupedLinear class
class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ...
    ) -> torch.Tensor:
        ...

        # Initialize weights
        weights_fp8: list
        if fp8:
          ...
        else:
            # Change this section
            # Check environment variable to apply INT4 fake quantization
            if os.getenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0") == "1":
                group_size = int(os.getenv("OPEN_TRAINING_INT4_GROUP_SIZE", "128"))
                weights_fp8 = [fake_int4_quantization_ste(w, [1, group_size]) for w in weights]
            else:
                weights_fp8 = [cast_if_needed(weight, activation_dtype) for weight in weights]
```

#### 4. Start INT4 Training

You need to configure the specific environment variables for quantization settings.

**Environment Variables:**

*   **`OPEN_TRAINING_INT4_FAKE_QAT_FLAG`**: Enables fake quantization operations for INT4 training.
*   **`OPEN_TRAINING_INT4_GROUP_SIZE`**: Specifies the block size (group size) for model quantization.
    *   Set to **128** for `moonlight-16B-A3B` 、 `qwen3-30B-A3B`and `qwen3-235B-A22B-int4`.
    *   Set to **32** for `kimi-k2-Thinking-int4`.

**Configuration Example:**

```json
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    ...
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"128\"
  }
}"
```

**Launch Commands:**

```bash
# Moonlight-16B-A3B Int4 training
bash examples/low_precision/run-moonlight-16B-A3B-int4.sh

# Qwen3‑30B‑A3B Int4 training
bash examples/low_precision/run-qwen3‑30B‑A3B-int4.sh

# Qwen3-235B-A22B Int4 training (8 nodes)
bash examples/low_precision/run-qwen3-235B-A22B-int4.sh

# Kimi-k2-Thinking Int4 training (32 nodes)
bash examples/low_precision/run-kimi-k2-Thinking-int4.sh
```

- For multi-node environments, please start the Ray service according to your cluster configuration.