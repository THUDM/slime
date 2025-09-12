# SFT training of Qwen3 models

This example demonstrates the correct sft loss mask computation logic when training Qwen3 models, especially the **Qwen3 Thinking series models** (e.g., **Qwen3-4B-Thinking-2507**).

## Files

- `sft_rollout.py`: The correct sft loss mask computation logic of Qwen3 models

## Usage

1. Setup and download model:
```bash
cd slime
pip install -e .

hf download Qwen/Qwen3-4B-Thinking-2507 --local-dir /root/Qwen/Qwen3-4B-Thinking-2507
```

2. Create torch dict
```bash
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen/Qwen3-4B-Thinking-2507 \
    --rotary-base 5000000 \
    --save /root/Qwen/Qwen3-4B-Thinking-2507_torch_dist
```

3. SFT:
```bash
bash examples/qwen3_thinking_sft/run-qwen3-4b-thinking-sft.sh
```
