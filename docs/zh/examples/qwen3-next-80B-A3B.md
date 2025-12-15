# 8xH100 训练 Qwen3-30B-A3B

## 环境准备

搭建环境、下载模型、数据与 ckpt 转换均与 Qwen3-4B 模型相同，可以参考 [示例：Qwen3-4B](./qwen3-4B.md)，将文中 Qwen3-4B 的部分转换为 Qwen3-next-80B-A3B-Instruct 即可。

可以用如下完整方法把 huggingface checkpoint 转化为 torch_dist 格式：

```bash
# 下载模型权重 (Qwen3-Next-80B-A3B-Thinking)
hf download Qwen/Qwen3-Next-80B-A3B-Thinking --local-dir /root/Qwen3-Next-80B-A3B-Thinking
```

```bash
cd slime/
pip install -e .
source scripts/models/qwen3-next-80B-A3B.sh
PYTHONPATH=/root/Megatron-LM/ torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-Next-80B-A3B-Thinking/ \
   --save /root/Qwen3-Next-80B-A3B-Thinking_torch_dist/
```

## 执行训练 (Megatron)

执行训练：

```bash
cd /root/slime
export BASE_FOLDER=/root
export MASTER_ADDR=127.0.0.1
bash scripts/run-qwen3-Next-80B-A3B.sh 
```

## 执行训练 (FSDP)