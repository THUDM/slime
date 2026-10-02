# Reproducibility

Reproducibility is a bedrock of scientific progress. By combining [deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) from SGLang with Megatron-LM deterministic mode, slime can provide bitwise experiment reproduction.

To enable deterministic training, uninstall FlashAttention 3 with `pip uninstall flash_attn_3 -y`, then set:

```bash
  # sglang config
  --sglang-enable-deterministic-inference
  --sglang-attention-backend flashinfer

  # megatron config
  --deterministic-mode
```

Also set the following environment variables:

```bash
     "env_vars": {
        ...,
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
```

We provide a fully deterministic GSM8K training script for Qwen2.5-0.5B.

Use the following commands to initialize the training data and checkpoint:

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

Run training with:

```bash
bash scripts/run-qwen2.5-0.5B-reproducibility.sh
```

The wandb screenshots are recorded in [pull#370](https://github.com/THUDM/slime/pull/370).

## Train/rollout log-prob alignment (GLM-5)

Beyond single-side bitwise reproduction, slime can align the training log-probs with the rollout (inference) log-probs. This is currently supported only for the **GLM-5 structure** (MLA + DSA sparse attention), and requires the deterministic SGLang / batch-invariant DeepGEMM / DeepEP build plus the `megatron-sglang-aligned.patch` Megatron.

Supported in this path:

- DSA sparse attention (`flashmla_sparse` prefill/decode), including deterministic NSA RadixCache/prefix cache;
- DeepGEMM batch-invariant block-FP8 forward for dense and grouped-MoE layers (with BF16 backward);
- fp32 MoE router (the LM head stays bf16 on both train and rollout — matching precision, not fp32, is what aligns);
- SGLang DeepEP low-latency rollout plus Megatron DeepEP normal training. A
  compact second normal dispatch preserves every top-k route, and the token
  owner performs the weighted reduction in slot order and FP32. Ordinary
  Megatron all-to-all is not an alignment backend for this path;
- bf16 or FP8-E4M3 KV cache. For `flashmla_sparse`, SGLang stores packed FP8
  cache entries and gathers/dequantizes only the selected pages before its BF16
  sparse kernel. The maintained gate defaults to FP8-E4M3 and does not use
  rollout routing replay (R3), so all main-model parameters, including the
  router and experts, execute backward. The auxiliary DSA indexer remains
  frozen through `--freeze-indexer`.

The regression gate is `tests/test_glm52_6layer_deterministic_e2e.py` (6-layer GLM-5.2, single-node EP8): it runs a real Megatron→SGLang online-weight-update rollout, trains all main-model parameters, and asserts `train_rollout_logprob_abs_diff < 1e-6` (the established DeepEP alignment reference is in the `x e-7` range).

An additional short EP8 gate, `tests/test_glm52_layerwise_zero_e2e.py`, records
the visible output of decoder layers 0–5 on both sides and requires every
matched hidden-state element to have an absolute difference of exactly zero.
