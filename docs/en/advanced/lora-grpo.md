# Megatron-Bridge LoRA for GRPO

slime supports a first Megatron-Bridge LoRA path for dense GRPO actor training. This path keeps training in Megatron, merges LoRA adapters only during SGLang weight export, and restores the unmerged actor weights immediately after export.

## Example

Start from a dense Megatron GRPO script such as `scripts/run-qwen3-4B.sh`, then add the LoRA and bridge flags:

```bash
--enable-lora \
--megatron-to-hf-mode bridge \
--colocate \
--lora-rank 16 \
--lora-alpha 32 \
--lora-dropout 0.0 \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
```

`--lora-target-modules` is optional. If it is omitted, slime uses the Megatron-Bridge LoRA defaults.

## Supported Scope

The initial LoRA path intentionally supports a narrow, validated configuration:

- Megatron training backend.
- Megatron-Bridge HF weight export mode.
- GRPO actor training.
- Colocated SGLang rollout and weight updates.
- Dense models.
- Default weight backuper enabled.

The following combinations are rejected at startup until they have dedicated parity coverage:

- MoE models.
- PPO or critic-based training.
- Decoupled rollout mode outside `--debug-train-only`.
- Custom model providers.
- `--only-train-params-name-list` or `--freeze-params-name-list`.
- On-policy distillation.
- Reference model update intervals.
- `--disable-weights-backuper`.

