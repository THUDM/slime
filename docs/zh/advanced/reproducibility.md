# Reproducibility

Reproducibility is a bedrock of scientific progress. By combining the [deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) of SGLang and the deterministic mode of Megatron-LM, slime supports bitwise experiment reproduction.

To enable deterministic training, you need to first uninstall the flash attention 3 in the docker with `pip uninstall flash_attn_3 -y` and set:
```bash
  # sglang config
  --sglang-enable-deterministic-inference
  --sglang-attention-backend flashinfer

  # megatron config
  --deterministic-mode
```

And set the following environment variables:

```bash
     "env_vars": {
        ...,
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
```

GSM8K 样例使用 Qwen3-4B-Instruct-2507，启用 deterministic、CP=4 和 128K 回复上限。

For data and checkpoint preparation, please run:

```bash
# download
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507

# convert ckpt
cd slime/
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM/ python \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
   --save /root/Qwen3-4B-Instruct-2507_torch_dist/
```

And to run training,

```bash
bash scripts/run-qwen3-4B-reproducibility.sh
```

For screen shots of the wandb, please refer to [pull#370](https://github.com/THUDM/slime/pull/370).

## Train/rollout log-prob alignment (GLM-5)

Beyond single-side bitwise reproduction, slime can align the training log-probs with the rollout (inference) log-probs. This is currently supported only for the **GLM-5 structure** (MLA + DSA sparse attention), and requires the deterministic SGLang / batch-invariant DeepGEMM / DeepEP build plus the `megatron-sglang-aligned.patch` Megatron.

Supported in this path:

- DSA sparse attention (`flashmla_sparse` prefill/decode), including deterministic NSA RadixCache/prefix cache;
- DeepGEMM batch-invariant block-FP8 forward for dense and grouped-MoE layers (with BF16 backward);
- fp32 MoE router (the LM head stays bf16 on both train and rollout — matching precision, not fp32, is what aligns);
- SGLang rollout 使用 DeepEP low-latency，Megatron 训练使用 DeepEP normal。
  第二次小 payload normal dispatch 保留每个 top-k route，token owner 按
  slot 顺序做 FP32 加权归约；这条对齐路径不支持普通 Megatron all-to-all；
- 支持 bf16 或 FP8-E4M3 KV cache。`flashmla_sparse` 路径把 KV 以 FP8
  packed 格式保存，只 gather 并反量化被选中的 page，再交给 BF16 sparse
  kernel。维护的 gate 默认使用 FP8-E4M3，不使用 rollout routing replay
  (R3)，因此包括 router 和 experts 在内的主模型参数都会执行 backward；
  辅助 DSA indexer 通过 `--freeze-indexer` 始终保持冻结。

回归 gate 是 `tests/test_glm52_6layer_deterministic_e2e.py`（6-layer GLM-5.2，
单机 EP8）：它执行真实的 Megatron→SGLang online-weight-update rollout，
训练全部主模型参数，并断言 `train_rollout_logprob_abs_diff < 1e-6`（已验证的
DeepEP 对齐参考结果为 `x e-7` 量级）。

另有一个较短的 EP8 gate `tests/test_glm52_layerwise_zero_e2e.py`，会同时
记录训推两侧 decoder layer 0–5 的可见输出，并要求所有匹配 hidden-state
元素的绝对误差严格等于 0。

### DeepGEMM 训练前向的 FP8 权重缓存

使用自定义 DeepGEMM 前向时，可添加 `--megatron-deepgemm-cache-weights`
（默认关闭）。该开关不负责安装 DeepGEMM hook；仍需配置所需层和 hook。

开启后，Dense Linear 和 grouped MoE 的权重在每个训练 step 的
forward/backward 调度范围内按需量化一次，供后续 microbatch 和 activation
checkpoint 重算复用。缓存只包含 FP8 权重及 scale，不缓存激活。同一调度中
权重版本、存储位置或布局改变会重新量化；MoE 按 expert 分组保存缓存。

缓存会在进入 optimizer 更新前释放，异常退出也会释放。每次 logprob/value
前向评估使用独立缓存，结束后释放，避免 actor/ref/old-policy 切换时复用旧值。
它不依赖优化器是否更新 Tensor 的版本计数，也不跨训练 step 持久保存。
在标准调度之外直接调用低层函数时，可用
`deepgemm_weight_cache_scope(True)` 显式限定权重不变的范围；不要在这个范围内
通过 `.data` 原地改写权重。

开关开启会增加存活 FP8 权重（约每参数 1 字节）及 scale 的显存占用；MoE
缓存拥有独立存储，不再将缓存内容放入随后被激活覆盖的临时工作区。关闭时
保留原有每次前向量化和工作区复用行为。它不改变激活量化、前向精度或反向规则。
