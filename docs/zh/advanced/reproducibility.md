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

Here we provide the script to do RL training on Qwen2.5 0.5B model and GSM8K dataset with full deterministic.

For data and checkpoint preparation, please run:

```bash
# download
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# convert ckpt
cd slime/
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM/ python \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
   --save /root/Qwen2.5-0.5B-Instruct_torch_dist/
```

And to run training,

```bash
bash scripts/run-qwen2.5-0.5B-reproducibility.sh
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
