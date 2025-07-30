# slime

[English](./README.md)

**slime** 是为 RL scaling 设计的 LLM post‑training 框架，提供两大核心能力：

1. **高性能训练**：通过连接 Megatron 与 SGLang，支持各种模式的高效训练；
2. **灵活的数据生成**：通过自定义数据生成接口以及 server based engine，实现任意的数据训练数据生成流程。

## 目录

- [架构总览](#架构总览)
- [快速开始](#快速开始)
  - [环境准备](#环境准备)
  - [示例](#示例)
    - [Dense 模型示例：GLM-4-9B 与 Qwen3-4B](#Dense-模型示例GLM-4-9B-与-Qwen3-4B)
    - [MoE 模型示例：Qwen3-30B-A3B 与 DeepSeek-R1](#MoE-模型示例Qwen3-30B-A3B-与-DeepSeek-R1)
    - [多轮对话 + 工具调用示例：Search-R1 lite](#多轮对话--工具调用示例Search-R1-lite)
    - [SFT 示例：Qwen3-4B-Base + OpenHermes-2.5](#SFT-示例Qwen3-4B-Base--OpenHermes-25)
- [Checkpoint 格式转换](#checkpoint-格式转换)
- [启动训练流程](#启动训练流程)
- [参数说明](#参数说明)
- [开发指南](#开发指南)
- [常见 Q&A 与致谢](#常见-qa-与致谢)

## 架构总览

![arch](./imgs/arch.png)

**模块说明**：

- **training (Megatron)**：负责主训练流程，从 Data Buffer 读取数据，训练完后将参数同步至 rollout 模块；
- **rollout (SGLang + router)**：生成新数据（含 reward/verifier），存储至 Data Buffer；
- **data buffer**：桥梁模块，管理 prompt 初始化、自定义数据与 rollout 生成方法。

## 快速开始

### 环境准备

基于镜像 zhuzilin/slime:latest（已预装 SGLang 0.4.7 和 Megatron）：

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it zhuzilin/slime:latest /bin/bash

git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
```

- 对于不方便使用 docker 的场景，请参考 [从零搭建环境](./docs/zh/build.md)；
- 对于 AMD 支持，请参考 [AMD 使用教程](./docs/en/amd_tutorial.md)。

### 示例

#### Dense 模型示例：GLM-4-9B 与 Qwen3-4B

我们提供了 [GLM-4-9B](https://huggingface.co/THUDM/GLM-Z1-9B-0414) 和 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) 的使用示例，可以通过他们对 slime 的使用方法有个基本的了解：

- [示例：GLM-4-9B](docs/zh/models/glm4-9B.md)
- [示例：Qwen3-4B](docs/zh/models/qwen3-4B.md)

#### MoE 模型示例：Qwen3-30B-A3B 与 DeepSeek-R1

我们也提供了 MoE 模型的示例，请查看：

- [示例：Qwen3-30B-A3B](docs/zh/models/qwen3-30B-A3B.md)
- [示例：128xH100 训练 DeepSeek-R1](docs/zh/models/deepseek-r1.md)

#### 多轮对话 + 工具调用示例：Search-R1 lite

针对多轮对话和工具调用场景，我们提供了一个简化版的 Search-R1 复现，请查看：

- [示例：Search-R1 lite](examples/search-r1/README_zh.md)

#### SFT 示例：Qwen3-4B-Base + OpenHermes-2.5

slime is not just a RL framework, we support a diverse set of post-training setups. For an SFT example, please refer to:

slime 不仅仅是一个 RL 框架，我们还支持了各种后训练流程。如果想使用 SFT，请参看：

- [示例: Qwen3-4B-Base + OpenHermes-2.5](docs/zh/sft.md).

### Checkpoint 格式转换

由于 slime 使用 megatron，而 megatron 不支持加载 huggingface checkpoint，我们需要将模型转换至 megatron 可以支持的 torch_dist 格式。

#### HF → Megatron torch_dist ckpt

我们推荐使用 [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) 进行转换。如果你目前在使用的模型不被 Pai-Megatron-Patch 支持，可以使用 [mbridge](https://github.com/ISEEKYAN/mbridge.git) 转换：

```bash
cd slime/
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist
```

⚠️  如果出现找不到 slime 的问题，请在 slime 目录下 `pip install -e .`。

#### Megatron torch_dist → HF ckpt

将训练过程中的存储的 torch_dist ckpt 转为 hf ckpt：

```bash
cd slime/
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/torch_dist_ckpt/iter_xxx/ \
  --output-dir /root/GLM-Z1-9B-0414-iter_xxx \
  --origin-hf-dir /root/GLM-Z1-9B-0414
```

⚠️ 由于 mbridge 转换的 torch_dist ckpt 目前不保存 args，不能基于上一步的 torch_dist ckpt 反转回 HF。

#### 任意 Megatron ckpt → HF

适用于自定义保存格式（如 `--ckpt-format torch`）。

转化方式的原理是直接复用训练中，从 megatron 向 sglang 更新参数的函数，也就是直接复用一下训练脚本，将原先的：

```bash
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": { ...}
   }' \
   -- python3 train.py \
   ... # 其他训练 args
```

改成：

```bash
torchrun --nproc_per_node ${NUM_GPU} tools/convert_to_hf.py \
   --load /your/saved/megatron_ckpt \
   --output-dir /your/converted/hf_ckpt \
   ... # 其他训练 args
```

即，保持所有的参数不变，将：

1. 任务启动从 ray 变成 torchrun，把 gpu 数量保存为 megatron 并行的不带 dp 的最小 gpu 数，例如如果是 tp4，就设成 4；
2. 确认把 `--load` 改成了需要 load 的路径；
3. 增加 `--output-dir` 对应要保存的 hf_ckpt。

## 启动训练流程

整个程序需要使用 ray 进行启动，首先需要启动一个 ray 集群，即在 node 0 运行：

```bash
# Node0（HEAD）
ray start --head --node-ip-address ${MASTER_ADDR} \
  --num-gpus 8 --disable-usage-stats

# 其他 Node
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8
```

在 ray 集群启动后，可以在 node 0 提交任务，例如：

```bash
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        ... # e.g. no_proxy、接口变量等
     }
   }' \
   -- python3 train.py \
   --...（其他 Megatron/SGLang/slime 参数）
```

#### 参数说明

参数分为三类：

1. **megatron 参数**：slime 会读取 `PYTHONPATH` 中的 megatron 里设置的所有参数，可以通过传入如 `--tensor-model-parallel-size 2` 的方式配置 megatron；
2. **sglang 参数**：支持环境中安装的 sglang 的所有参数，这些参数需要以 `--sglang` 起始，例如 `--mem-fraction-static` 需要通过 `--sglang-mem-fraction-static` 传入。
3. **slime 自身的参数**：请见：[slime/utils/arguments.py](slime/utils/arguments.py)

完整使用说明请查阅 [使用文档](docs/zh/usage.md)。

## 开发指南

- **欢迎贡献！** 若有功能建议、性能调优或使用体验反馈，欢迎提交 Issue / PR 😊

- 使用 [pre-commit](https://pre-commit.com/) 保证提交代码风格：

  ```bash
  apt install pre-commit -y
  pre-commit install
  ```

- 调试技巧请参考 [debug 指南](docs/zh/debug.md)

## 常见 Q&A 与致谢

- 常见问题请见 [Q&A](docs/zh/qa.md)
- 特别感谢以下项目 & 社区：SGLang、Megatron‑LM、mbridge、OpenRLHF、veRL 等。
