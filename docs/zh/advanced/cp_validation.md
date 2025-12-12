# 上下文并行（CP）优化验证流程

本文给出针对 [issue #1062](https://github.com/THUDM/slime/issues/1062) 的通信优化的完整验证方案，重点关注长序列下的 CP 训练迭代时间。流程覆盖环境准备、硬件需求、最小可复现用例及结果判定，便于在小模型上快速确认架构与 CUDA 支持是否正常。

## 环境与硬件要求
- **GPU 数量**：CP=2 需要至少 2 张 GPU；建议同一节点内进行以降低通信开销。若要验证 CP=4，则需 4 张 GPU。
- **显存**：单卡 24GB 即可跑小模型 smoke test；显存越大可提升序列长度上限。
- **GPU 架构**：Ampere 及以上（SM80+/bf16 支持）。消费级 5090 只要驱动与 CUDA 版本满足下述要求即可验证。
- **驱动/CUDA/NCCL**：推荐驱动 ≥ 550，CUDA 12.1/12.2（FA3 需 >=12.1），NCCL ≥ 2.18。保持所有 GPU 使用相同驱动版本。
- **网络**：单机 PCIe 可用；多机验证需确保 NCCL 能连通（`NCCL_IB_DISABLE=1` 可在无 IB 时落到 TCP）。

## 软件准备
1. 安装依赖
   ```bash
   conda create -n slime-cp python=3.10 -y
   conda activate slime-cp
   pip install -r requirements.txt
   pip install -e .
   ```
2. 获取最小模型（示例使用 HuggingFace `Qwen/Qwen2.5-0.5B-Instruct`，体积小便于 smoke test）：
   ```bash
   mkdir -p /root/models
   huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/models/Qwen2.5-0.5B-Instruct
   ```
3. 准备极简提示集（8 条即可覆盖前向/反向与 KL、entropy 分支）：
   ```bash
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

## 验证步骤
1. **静态检查**：确保 Megatron 相关模块可编译。
   ```bash
   python -m compileall slime/backends/megatron_utils
   ```
2. **运行基线（CP=1）**：单节点 2 卡，关闭 CP，记录单 step 耗时与日志。
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN \
   python train.py \
     --train-backend megatron \
     --actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate \
     --hf-checkpoint /root/models/Qwen2.5-0.5B-Instruct \
     --ref-load /root/models/Qwen2.5-0.5B-Instruct \
     --prompt-data /root/datasets/cp_smoke.jsonl --input-key messages --apply-chat-template --rollout-shuffle \
     --rollout-num 8 --rollout-batch-size 2 --n-samples-per-prompt 1 --rollout-max-response-len 1024 \
     --global-batch-size 8 \
     --advantage-estimator gspo --use-kl-loss --kl-loss-coef 0.0 --kl-loss-type low_var_kl --entropy-coef 0.0 \
     --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
     --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --sequence-parallel --context-parallel-size 1 \
     --use-dynamic-batch-size --max-tokens-per-gpu 2048 \
     --rollout-num-gpus-per-engine 1 \
     --ci-test --ci-disable-kl-checker
   ```
   关键观察：
   - 日志中的 `iteration time` 或 `throughput`，保留 3~5 个 step 的平均值。
   - 确认无额外 all-gather/collective 报错。

3. **开启 CP=2**：仅改 `--context-parallel-size 2`，其余参数保持，使用同样的 2 张 GPU。
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN \
   python train.py \
     --train-backend megatron \
     --actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate \
     --hf-checkpoint /root/models/Qwen2.5-0.5B-Instruct \
     --ref-load /root/models/Qwen2.5-0.5B-Instruct \
     --prompt-data /root/datasets/cp_smoke.jsonl --input-key messages --apply-chat-template --rollout-shuffle \
     --rollout-num 8 --rollout-batch-size 2 --n-samples-per-prompt 1 --rollout-max-response-len 1024 \
     --global-batch-size 8 \
     --advantage-estimator gspo --use-kl-loss --kl-loss-coef 0.0 --kl-loss-type low_var_kl --entropy-coef 0.0 \
     --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
     --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --sequence-parallel --context-parallel-size 2 \
     --use-dynamic-batch-size --max-tokens-per-gpu 2048 \
     --rollout-num-gpus-per-engine 1 \
     --ci-test --ci-disable-kl-checker
   ```
   关键观察：
   - 与 CP=1 相比，单 step 耗时应接近 1 倍（不再出现 2 倍放大）。
   - KL/entropy 相关日志不应出现新的 all-gather 或形状不匹配错误。

4. **结果判定与常见问题**
   - 若 CP=2 仍明显慢于 CP=1，检查驱动/NCCL 版本是否过旧、是否落到 TCP、或是否开启了不支持的 FA3 组合（可暂时 `--attention-backend torch` 验证）。
   - 如遇 5090 等消费卡，确认驱动版本支持 CUDA 12.1+，并关闭未适配的 MIG/功耗限制；必要时设置 `NCCL_P2P_DISABLE=1` 观察差异。
   - 可通过 `NCCL_DEBUG=INFO`、`TORCH_DISTRIBUTED_DEBUG=DETAIL` 收集额外日志，确认新的 CP 掩码/归约路径是否稳定。

完成以上 3 步可在极小模型上覆盖 GSPO/OPSM 相关的 KL、entropy 与本地掩码逻辑，验证本次通信优化在长序列 CP 场景下的正确性与性能收益。
