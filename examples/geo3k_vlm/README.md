# FSDP + VLM Single-Turn RL

Training VLMs with FSDP on single-turn reasoning task using GRPO on the [GEO3K dataset](https://huggingface.co/datasets/hiyouga/geometry3k). We used processed version [here](https://huggingface.co/datasets/chenhegu/geo3k_imgurl).

<p align="center">
  <img src="rewards.png" alt="Reward Plot" width="800">
</p>

## Reproduce

```bash
export WANDB_API_KEY=your_wandb_api_key

SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-2B-Instruct SLIME_SCRIPT_NUM_GPUS=8 python examples/geo3k_vlm/run_geo3k_vlm.py 2>&1 | tee run_simple.log
```

## Notes

We experimented with three reward model configurations:
1. A geo3k-specific RM with tolerance=0.05 (to handle rounding in ground truth labels)
2. A geo3k-specific RM with tolerance=0.0 (strict matching)
3. The default math RM

All three configurations performed similarly, so we use the default math RM for simplicity.