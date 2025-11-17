
```bash
cd /root/workspace/slime-open/examples/search-r1-oai
python data_preprocess.py

# hf checkpoint
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /root/workspace/Qwen2.5-3B-Instruct

# mcore checkpoint
cd /root/workspace/slime-open
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
     ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/workspace/Qwen2.5-3B-Instruct \
    --save /root/workspace/Qwen2.5-3B-Instruct_torch_dist
```