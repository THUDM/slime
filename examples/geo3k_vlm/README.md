# VLM Single-Turn RL

Training VLMs with Megatron on single-turn reasoning task using GRPO on the [GEO3K dataset](https://huggingface.co/datasets/hiyouga/geometry3k). We used processed version [here](https://huggingface.co/datasets/chenhegu/geo3k_imgurl).

Supported models:
* Qwen3.5-35B-A3B
* NVIDIA Nemotron 3.5 Super VL

Note: Please make sure the cudnn version in the environment is 9.16.0.29 to prevent severe performance regression in conv3d in torch 2.9 mentioned in https://github.com/pytorch/pytorch/issues/168167. Otherwise, you can reinstall cudnn with:
```bash
pip install nvidia-cudnn-cu12==9.16.0.29
```

Qwen3.5 uses slime's native Megatron language model plus the Transformers vision model. Vision parameters retain their HuggingFace names and are loaded and synchronized directly.

Nemotron 3.5 Super VL (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`) uses MCore's `HybridModel` (Mamba2, MoE and attention layers) as the language model, plus the Transformers (remote-code) RADIO vision tower and projector. Its vision parameters are renamed when they are loaded and when they are synchronized: the checkpoint's fused `attn.qkv` is split into query/key/value for the Transformers module and fused again when syncing to SGLang (whose `QKVParallelLinear` accepts only the fused form), and the projector's `mlp1.*` keys are mapped to `vision_projector.*`. The vision tower's LayerScale is removed on both sides: C-RADIO ships no LayerScale weights and `layerscale_value` is 1.0, so it would only be an identity.

<p align="center">
  <img src="fsdp_vs_megatron.png" alt="Reward Plot" width="800">
</p>

## Data Preparation (For SFT Training)

The [geo3k_imgurl](https://huggingface.co/datasets/chenhegu/geo3k_imgurl) dataset contains:
- `problem`: The math problem text (string)
- `answer`: The answer (string, e.g., "270")
- `images`: Image data (list)

For SFT training, we need to format the `answer` field for `\boxed{}` format and the messages. You can use the following script to format the answer field:

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("chenhegu/geo3k_imgurl", split="train")

def format_answer(answer: str) -> str:
    """Format answer to include \\boxed{} format."""
    return f"Answer: \\boxed{{{answer}}}"

def process_sample(sample):
    formatted_answer = f"Answer: \\boxed{{{sample['answer']}}}"
    
    sample["messages"] = [
        {"role": "user", "content": sample["problem"]},
        {"role": "assistant", "content": formatted_answer}
    ]
    return sample

ds = ds.map(process_sample)
ds.to_parquet("/root/datasets/geo3k_imgurl/train_formatted.parquet")
```

## Reproduce

```bash
export WANDB_API_KEY=your_wandb_api_key

./examples/geo3k_vlm/run_geo3k_qwen35.sh
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SLIME_SCRIPT_DATASET_NAME` | `chenhegu/geo3k_imgurl` | HuggingFace dataset name |
| `SLIME_SCRIPT_NUM_GPUS` | `8` | Number of GPUs |
| `SLIME_SCRIPT_EXTERNAL_RAY` | `0` | Use external Ray cluster (`1` to enable) |

#### Qwen3.5 Series
We provide a native [example](./run_geo3k_qwen35.sh) for Qwen3.5-35B-A3B. It loads the Transformers ViT directly and converts only the Megatron language-model parameters. To support another Qwen3.5 model, add its model config under `scripts/models/` and update the example.

The native path supports tensor, pipeline, context, sequence, and expert parallelism. Context parallelism follows Megatron's packed THD zigzag layout.

For GDN training, use `--micro-batch-size 1` and remove `--use-dynamic-batch-size`.

#### Nemotron 3.5 Series
Compared with `run_geo3k_qwen35.sh`, a Nemotron 3.5 Super VL run needs the following (the training hyperparameters can stay the same):

* **Model config and spec.** Pass `--spec slime_plugins.models.nemotron_h_vl get_nemotron_h_vl_model_provider` together with the model config `scripts/models/nemotron3.5-super-120B-A12B.sh`.
* **Four nodes, expert parallelism 32.** The model has 512 routed experts, and EP x ETP must divide the number of GPUs, so use `--expert-model-parallel-size 32` on 32 GPUs (TP2 x CP2, DP8). A single node of 8 GPUs fits only the small proxy model. A multi-node run needs a Ray cluster that already spans the nodes, not the script's own one-node `ray start --head`.
* **`mamba-ssm` and `causal-conv1d` must be installed.** MCore's `MambaMixer` requires them for the Mamba2 layers. They replace `pip install -U transformers`, and the `docker/Dockerfile` image builds them. The SGLang side needs the patches in `docker/patch/*/` (`sglang-radio_layerscale.patch`, `sglang-nemotron-multi-image.patch`), which the image applies.
* **Offload the optimizer states** (`--optimizer-cpu-offload --use-precision-aware-optimizer` on Megatron 0.20) to fit 80 GB GPUs. slime parses Megatron's arguments with `ignore_unknown_args=True`, so a flag spelling the installed Megatron does not have is dropped without an error and the offload silently stops happening.
* **MTP.** The checkpoint has one MTP layer, used by SGLang as the EAGLE draft and trained by Megatron: `--mtp-num-layers N --mtp-use-repeated-layer --enable-mtp-training --mtp-loss-scaling-factor 0.2`, with one `/*E` segment per depth appended to `--hybrid-override-pattern`, and `--sglang-speculative-algorithm EAGLE` with `--sglang-speculative-moe-runner-backend triton`. The weight sync keeps the draft on the trained layer.
* **No Qwen-specific SGLang pins.** `--sglang-moe-runner-backend triton` and `--sglang-mm-attention-backend sdpa` are for Qwen's gated experts and vision tower; Nemotron's experts are not gated (squared ReLU, no `gate_proj`) and its vision tower is served by SGLang's own `radio.py`.

For more details, see `tests/test_nemotron35_super_vl_proxy_e2e.py`, which runs the same GRPO path on a random-weight proxy of the model on one node of 8 GPUs.

## Notes

### Reward Model Configuration

We experimented with three reward model configurations:
1. A geo3k-specific RM with tolerance=0.05 (to handle rounding in ground truth labels)
2. A geo3k-specific RM with tolerance=0.0 (strict matching)
3. The default math RM

All three performed similarly, so we use the default math RM for simplicity.

### Numerical Precision with Non-Binary Rewards

Our initial geo3k-specific verifier produced "format scores" (**0 and 0.9**) instead of clean binary rewards. Under **fp32**, fractional values like 0.9 can't be exactly represented, so when all samples in a group have the same reward, `reward - mean` doesn't equal zero—creating spurious gradient signal.

We fixed this by switching to the default math RM with clean **binary 0/1 rewards**. If you encounter similar precision issues with non-binary rewards, you can change the reward tensor dtype from `torch.float` to `torch.float16` in `slime/ray/rollout.py` (`_post_process_rewards` method) to truncate precision artifacts.

## B200
Blackwell currently does not support fa3, we need to use  `--sglang-mm-attention-backend sdpa` and `--attn-implementation flash_attention_2`
