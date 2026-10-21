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

The GSM8K example enables deterministic training for Qwen3-4B-Instruct-2507, with CP=4 and a 128K response limit.

Use the following commands to initialize the training data and checkpoint:

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

Run training with:

```bash
bash scripts/run-qwen3-4B-reproducibility.sh
```

Historical Qwen2.5-0.5B wandb screenshots are recorded in [pull#370](https://github.com/THUDM/slime/pull/370).

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


### Cache FP8 weights in the DeepGEMM training forward

Add `--megatron-deepgemm-cache-weights` to reuse quantized dense and grouped-MoE
weights across microbatches and activation-checkpoint recomputation. The flag is
disabled by default and does not install the DeepGEMM hooks; configure the target
layers and hooks as usual.

Each training forward/backward schedule quantizes a weight or expert group on
first use. The cache is released before the optimizer update, including when an
optimizer writes through `.data` without incrementing the Parameter version.
Every logprob/value evaluation gets a separate cache, released at the end of the
call, so actor/reference/old-policy weight swaps cannot reuse stale values.
Version, storage, and layout changes within a schedule also invalidate entries.

Only FP8 weights and their scales are cached. Activation quantization and the
backward rules are unchanged. Cached MoE weights own their storage instead of
aliasing a temporary activation workspace. This increases live GPU memory by
roughly one byte per cached weight element plus scales; disabling the flag keeps
the existing repeated-quantization and workspace-reuse behavior.

Direct low-level calls outside the standard schedules remain uncached. For a
custom schedule, use `deepgemm_weight_cache_scope(True)` around the interval in
which weights stay unchanged. Do not mutate weights through `.data` inside that
scope.
