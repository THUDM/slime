# CP 验证流程

---

## 0) 一次性准备（环境 / 模型 / 数据）

```bash
set -euxo pipefail

# 0.1 Python 环境（按你仓库习惯可换成 uv/conda）
conda create -n slime-cp python=3.10 -y
conda activate slime-cp
pip install -r requirements.txt
pip install -e .

# 0.2 下载小模型（smoke test 足够快）
mkdir -p /root/models
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /root/models/Qwen2.5-0.5B-Instruct

# 0.3 准备极简 prompt 数据（8 条）
mkdir -p /root/datasets
cat > /root/datasets/cp_smoke.jsonl <<'DATA'
{"messages": [{"role": "user", "content": "证明 1+1=2 的最简单方法是什么？"}]}
{"messages": [{"role": "user", "content": "写一段 20 字以内的科幻故事"}]}
{"messages": [{"role": "user", "content": "概括勾股定理"}]}
{"messages": [{"role": "user", "content": "给出一道初中几何选择题并附答案"}]}
{"messages": [{"role": "user", "content": "用 3 句话介绍 transformer 的自注意力"}]}
{"messages": [{"role": "user", "content": "举例说明梯度累积的作用"}]}
{"messages": [{"role": "user", "content": "生成一条积极的日常鼓励语"}]}
{"messages": [{"role": "user", "content": "解释为什么要做归一化"}]}
DATA
```

---

## 1) 静态检查（可选但推荐）

```bash
python -m compileall slime/backends/megatron_utils
```

---

## 2) 跑基线：CP=1（2 卡单机）

```bash
set -euxo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN

MODEL=/root/models/Qwen2.5-0.5B-Instruct
DATA=/root/datasets/cp_smoke.jsonl

python train.py \
  --train-backend megatron \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate \
  --hf-checkpoint "$MODEL" \
  --ref-load "$MODEL" \
  --prompt-data "$DATA" --input-key messages --apply-chat-template --rollout-shuffle \
  --num-rollout 8 --rollout-batch-size 2 --n-samples-per-prompt 1 --rollout-max-response-len 1024 \
  --global-batch-size 8 \
  --advantage-estimator gspo --use-kl-loss --kl-loss-coef 0.0 --kl-loss-type low_var_kl --entropy-coef 0.0 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --sequence-parallel --context-parallel-size 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 2048 \
  --rollout-num-gpus-per-engine 1 \
  --ci-test --ci-disable-kl-checker
```

---

## 3) 开启 CP=2（只改一处：`--context-parallel-size 2`）

```bash
set -euxo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN

MODEL=/root/models/Qwen2.5-0.5B-Instruct
DATA=/root/datasets/cp_smoke.jsonl

python train.py \
  --train-backend megatron \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate \
  --hf-checkpoint "$MODEL" \
  --ref-load "$MODEL" \
  --prompt-data "$DATA" --input-key messages --apply-chat-template --rollout-shuffle \
  --num-rollout 8 --rollout-batch-size 2 --n-samples-per-prompt 1 --rollout-max-response-len 1024 \
  --global-batch-size 8 \
  --advantage-estimator gspo --use-kl-loss --kl-loss-coef 0.0 --kl-loss-type low_var_kl --entropy-coef 0.0 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --sequence-parallel --context-parallel-size 2 \
  --use-dynamic-batch-size --max-tokens-per-gpu 2048 \
  --rollout-num-gpus-per-engine 1 \
  --ci-test --ci-disable-kl-checker
```

---

## 4) 需要更详细 NCCL 日志时（可选）

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## 5) 你之前报错的根因（留给 issue 提出者用）

* ❌ 不要用：`--rollout-num 8`
* ✅ 改成：`--num-rollout 8`
* GPU 相关请用：`--rollout-num-gpus` 或 `--rollout-num-gpus-per-engine`
