# 示例：Qwen3-30B-A3B

[English](../../en/models/qwen3-30B-A3B.md)

## 环境准备

搭建环境、下载模型、数据与 ckpt 转换均与 Qwen3-4B 模型相同，可以参考 [示例：Qwen3-4B](./qwen3-4B.md)，将文中 Qwen3-4B 的部分转换为 Qwen3-30B-A3B 即可。

⚠️ 在做 MoE 模型的 ckpt 转换时，请使用 [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)，mbridge 可能会遇到加载问题。

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen3/run_8xH20.sh A3B /root/Qwen3-30B-A3B /root/Qwen3-30B-A3B_torch_dist false true bf16
```

[完整指令](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen3/README.md)

## 执行训练

执行训练：

```bash
cd /root/slime
bash scripts/run-qwen3-30B-A3B.sh
```

### 参数简介

这里我们简单介绍一下脚本 [run-qwen3-30B-A3B.sh](../../../scripts/run-qwen3-30B-A3B.sh) 中与 MoE 相关的部分。

1. 为了支持在 8xH800 环境中运行 Qwen3-30B-A3B，我们需要开启 megatron 的 CPU Adam 以节省显存，对应配置为：

   ```bash
   OPTIMIZER_ARGS=(
      ...
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
   ```

2. 开启 megatron 支持的 moe 优化，当前配置为 tp4, ep8：

   ```bash
   PERF_ARGS=(
      --tensor-model-parallel-size 4
      --sequence-parallel
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --expert-model-parallel-size 8
      --expert-tensor-parallel-size 1
      ...
   )
   ```

3. 开启 sglang 支持的 moe 优化，当前配置为 ep8：

   ```bash
   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 8
      --sglang-mem-fraction-static 0.5
      --sglang-enable-ep-moe
      --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   )
   ```

   类似地，也可以加入 dp attention，例如配置上：

   ```bash
      --sglang-enable-dp-attention
      --sglang-dp-size 8
   ```

### 多机支持

对于多机环境，需要进行如下的几点修改：
- 将训练模型，数据放在所有机器都可以访问到的路径上；
- 设置各台机器都可以访问到的 `MASTER_ADDR` 之外；
- 去掉 CPU adam 相关的配置，因为使用了 distributed optimizer，所以多机环境下 optimizer 的显存占比会明显下降。

除此之外，还可以进行如下的修改：

- 当总卡数并不能被 expert 总数乘除时，可以使用 `--sglang-ep-num-redundant-experts` 来增加冗余的 expert，例如对于 24 卡的场景，可以配置：

   ```bash
   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 24
      --sglang-mem-fraction-static 0.5
      --sglang-enable-ep-moe
      --sglang-enable-dp-attention
      --sglang-dp-size 3

      --sglang-moe-dense-tp-size 1
      --sglang-enable-dp-lm-head
      --sglang-ep-num-redundant-experts 16   
   )
   ```
