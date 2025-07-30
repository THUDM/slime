# slime

[中文版](./README_zh.md)

**slime** is an LLM post-training framework for RL scaling, providing two core capabilities:

1.  **High-Performance Training**: Supports efficient training in various modes by connecting Megatron with SGLang;
2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

## Table of Contents

  - [Architecture Overview](#architecture-overview)
  - [Quick Start](#quick-start)
      - [Environment Setup](#environment-setup)
      - [Examples](#examples)
        - [Dense Model Examples: GLM-4-9B and Qwen3-4B](#Dense-Model-Examples-GLM-4-9B-and-Qwen3-4B)
        - [MoE Model Example: Qwen3-30B-A3B and DeepSeek-R1](#MoE-Model-Example-Qwen3-30B-A3B-and-DeepSeek-R1)
        - [Multi-Turn + Tool Calling Example: Search-R1 lite](#Multi-Turn--Tool-Calling-Example-Search-R1-lite)
        - [SFT Example: Qwen3-4B-Base with OpenHermes-2.5](#SFT-Example-Qwen3-4B-Base-with-OpenHermes-25)
  - [Checkpoint Format Conversion](#checkpoint-format-conversion)
  - [Starting the Training Process](#starting-the-training-process)
  - [Argument Descriptions](#argument-descriptions)
  - [Developer Guide](#developer-guide)
  - [FAQ & Acknowledgements](#faq--acknowledgements)

## Architecture Overview

![arch](./imgs/arch.png)

**Module Descriptions**:

  - **training (Megatron)**: Responsible for the main training process, reads data from the Data Buffer, and synchronizes parameters to the rollout module after training.
  - **rollout (SGLang + router)**: Generates new data (including rewards/verifier outputs) and stores it in the Data Buffer.
  - **data buffer**: A bridge module that manages prompt initialization, custom data, and rollout generation methods.

## Quick Start

### Environment Setup

Based on the `zhuzilin/slime:latest` image (pre-installed with SGLang 0.4.7 and Megatron):

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it zhuzilin/slime:latest /bin/bash

git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
```

- If you prefer not to use Docker, or if it's inconvenient, please refer to [Setting up the Environment from Scratch](./docs/en/build.md).
- For AMD support, please refer to [AMD Tutorial](./docs/en/amd_tutorial.md).

### Examples

#### Dense Model Examples: GLM-4-9B and Qwen3-4B

We provide examples to use [GLM-4-9B](https://huggingface.co/THUDM/GLM-Z1-9B-0414) and [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), please refer to:

- [Example: GLM-4-9B](docs/en/models/glm4-9B.md).
- [Example: Qwen3-4B](docs/en/models/qwen3-4B.md).

#### MoE Model Example: Qwen3-30B-A3B and DeepSeek-R1

For MoE example, please refer to:

- [Example: Qwen3-30B-A3B](docs/en/models/qwen3-30B-A3B.md).
- [Example: Training DeepSeek R1 with 128xH100](docs/en/models/deepseek-r1.md)

#### Multi-Turn + Tool Calling Example: Search-R1 lite

For multi-turn and tool calling, we also provides an minimal reimplenmentation of Search-R1, please refer to:

- [Example: Search-R1 lite](examples/search-r1/README.md).

#### SFT Example: Qwen3-4B-Base with OpenHermes-2.5

slime is not just a RL framework, we support a diverse set of post-training setups. For an SFT example, please refer to:

- [Example: Qwen3-4B-Base with OpenHermes-2.5](docs/en/sft.md).

### Checkpoint Format Conversion

Since slime uses Megatron, and Megatron does not support loading Hugging Face checkpoints directly, we need to convert the model to the `torch_dist` format that Megatron supports.

#### HF → Megatron torch\_dist ckpt

We recommend using [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) for mcore checkpoint conversion.

If the mode you are using are not supported by Pai-Megatron-Patch, you could use [mbridge](https://github.com/ISEEKYAN/mbridge.git) for conversion:

```bash
cd slime/
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist
```

⚠️ If you encounter an issue where slime cannot be found, please run `pip install -e .` in the slime directory.

#### Megatron torch\_dist → HF ckpt

To convert a `torch_dist` checkpoint saved during training back to a Hugging Face checkpoint:

```bash
cd slime/
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/torch_dist_ckpt/iter_xxx/ \
  --output-dir /root/GLM-Z1-9B-0414-iter_xxx \
  --origin-hf-dir /root/GLM-Z1-9B-0414
```

⚠️ Since the `torch_dist` checkpoint converted by mbridge does not currently save args, you cannot convert the checkpoint from the previous step back to HF format.

#### Any Megatron ckpt → HF

Applicable for custom save formats (e.g., `--ckpt-format torch`).

The principle behind this conversion method is to reuse the function that updates parameters from Megatron to SGLang during training. This means reusing the training script and changing the original command from:

```bash
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": { ...}
   }' \
   -- python3 train.py \
   ... # Other training args
```

To:

```bash
torchrun --nproc_per_node ${NUM_GPU} tools/convert_to_hf.py \
   --load /your/saved/megatron_ckpt \
   --output-dir /your/converted/hf_ckpt \
   ... # Other training args
```

That is, keep all other arguments the same, and:

1.  Change the task launcher from `ray` to `torchrun`. Set the number of GPUs to the minimum required for Megatron's parallelism without data parallelism (DP). For example, if you are using `tp4`, set it to 4.
2.  Make sure to change `--load` to the path of the checkpoint you want to load.
3.  Add the `--output-dir` argument to specify where the converted Hugging Face checkpoint should be saved.

## Starting the Training Process

The entire program needs to be launched using Ray. First, you need to start a Ray cluster. On node 0, run:

```bash
# Node0 (HEAD)
ray start --head --node-ip-address ${MASTER_ADDR} \
  --num-gpus 8 --disable-usage-stats

# Other Nodes
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8
```

After the Ray cluster has started, you can submit a job from node 0, for example:

```bash
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        ... # e.g., no_proxy, API variables, etc.
     }
   }' \
   -- python3 train.py \
   --... # Other Megatron/SGLang/slime arguments
```

### Argument Descriptions

Arguments are divided into three categories:

1.  **Megatron arguments**: slime reads all arguments set in Megatron via `PYTHONPATH`. You can configure Megatron by passing arguments like `--tensor-model-parallel-size 2`.
2.  **SGLang arguments**: All arguments for the installed SGLang are supported. These arguments must be prefixed with `--sglang-`. For example, `--mem-fraction-static` should be passed as `--sglang-mem-fraction-static`.
3.  **slime-specific arguments**: Please refer to: [slime/utils/arguments.py](slime/utils/arguments.py)

For complete usage instructions, please refer to the [Usage Documentation](docs/en/usage.md).

## Developer Guide

  - **Contributions are welcome\!** If you have suggestions for new features, performance tuning, or feedback on user experience, feel free to submit an Issue or PR 😊

  - Use [pre-commit](https://pre-commit.com/) to ensure code style consistency for your commits:

    ```bash
    apt install pre-commit -y
    pre-commit install
    ```

  - For debugging tips, please refer to the [Debugging Guide](docs/en/debug.md)

## Hardware Support
- Nvidia: refer to this repo README
- AMD: refer to the [tutorial](docs/en/amd_tutorial.md)

## FAQ & Acknowledgements

  - For frequently asked questions, please see the [Q\&A](docs/en/qa.md)
  - Special thanks to the following projects & communities: SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, and others.
