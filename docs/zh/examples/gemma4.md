# Gemma4 Dense 与 MoE 的 GSM8K 示例

这个示例用于验证 Gemma4 text 模型在 slime 中的模型支持。这里使用
GSM8K，因为目标是验证 Megatron 模型路径、SGLang rollout 加载路径、loss
mask、反向传播和在线权重更新，不引入任务特定的 runtime 变量。

更大的任务特定 recipe 应当在这个验证通过后再接入。

## 运行内容

在单个 8 卡节点上分别运行 dense 和 MoE 版本：

| 模型 | 脚本 | Megatron 拓扑 | SGLang 拓扑 |
| --- | --- | --- | --- |
| `google/gemma-4-31B-it` | `scripts/run-gemma4-31B-gsm8k.sh` | TP2 PP4 CP1 | TP8 |
| `google/gemma-4-26B-A4B-it` | `scripts/run-gemma4-26B-A4B-gsm8k.sh` | TP2 PP2 EP2 CP1 | TP8 |

脚本默认只跑两个 rollout，并使用较短的 response length。它用于证明模型可以
完成训练闭环，不用于报告有意义的 GSM8K 分数。默认的一个很小的
`--entropy-coef` 用来确保在小样本全零 reward 时仍然会触发 optimizer 路径。

每种模型和拓扑都应使用新的转换 checkpoint 目录。默认路径包含 TP/PP/EP/CP，
因为 Megatron distributed checkpoint 会按转换拓扑切分。

## 准备 Checkpoint 与数据

```bash
cd /root
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e . --no-deps

hf download google/gemma-4-31B-it --local-dir /root/gemma-4-31B-it
hf download google/gemma-4-26B-A4B-it --local-dir /root/gemma-4-26B-A4B-it
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k
```

转换 dense checkpoint：

```bash
cd /root/slime
source scripts/models/gemma4-31B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint /root/gemma-4-31B-it \
   --tensor-model-parallel-size 2 \
   --pipeline-model-parallel-size 4 \
   --context-parallel-size 1 \
   --save /root/gemma-4-31B-it_tp2_pp4_cp1_torch_dist
```

转换 MoE checkpoint：

```bash
cd /root/slime
source scripts/models/gemma4-26B-A4B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint /root/gemma-4-26B-A4B-it \
   --tensor-model-parallel-size 2 \
   --pipeline-model-parallel-size 2 \
   --expert-model-parallel-size 2 \
   --context-parallel-size 1 \
   --save /root/gemma-4-26B-A4B-it_tp2_pp2_ep2_cp1_torch_dist
```

## 运行训练

```bash
cd /root/slime
bash scripts/run-gemma4-31B-gsm8k.sh
bash scripts/run-gemma4-26B-A4B-gsm8k.sh
```

如果需要记录到 W&B：

```bash
USE_WANDB=1 WANDB_PROJECT=slime-gemma4-gsm8k bash scripts/run-gemma4-31B-gsm8k.sh
USE_WANDB=1 WANDB_PROJECT=slime-gemma4-gsm8k bash scripts/run-gemma4-26B-A4B-gsm8k.sh
```

## 期望信号

成功运行时应当看到：

- SGLang 加载 `Gemma4ForConditionalGeneration`。
- 至少一个 rollout 和 train step 完成。
- stdout 或 W&B 中出现 `train/loss`、`train/grad_norm` 和 entropy 指标。
- Megatron 到 SGLang 的 raw `update_weights` 成功。

如果要做正式效果训练，应增加 rollout 数量、batch size、response length 和
eval interval，并设置 `ENTROPY_COEF=0`。
