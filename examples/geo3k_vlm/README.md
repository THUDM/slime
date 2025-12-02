# FSDP + VLM Single-Turn RL

Training VLMs with FSDP on single-turn reasoning task using GRPO on the [GEO3K dataset](https://huggingface.co/datasets/hiyouga/geometry3k). We used processed version [here](https://huggingface.co/datasets/chenhegu/geo3k_imgurl).

![Reward Plot](reward.png)

## Reproduce

```bash
export WANDB_API_KEY=your_wandb_api_key

SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-2B-Instruct SLIME_SCRIPT_EXTERNAL_RAY=1 SLIME_SCRIPT_NUM_GPUS=8 python examples/geo3k_vlm/run_geo3k_vlm.py 2>&1 | tee run_simple.log
```

## Notes

The GEO3K dataset contains some ground truth answers with rounding issues (e.g., exact answer `8/15` vs. ground truth `0.53`). The reward model uses a tolerance of 0.05 when comparing predicted answers to accommodate these discrepancies.