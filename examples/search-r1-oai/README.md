
```bash
python data_preprocess.py

# hf checkpoint
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /workspace/Qwen2.5-3B-Instruct

# mcore checkpoint
cd /workspace/slime-open
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
     ${MODEL_ARGS[@]} \
    --hf-checkpoint /workspace/Qwen2.5-3B-Instruct \
    --save /workspace/Qwen2.5-3B-Instruct_torch_dist
```