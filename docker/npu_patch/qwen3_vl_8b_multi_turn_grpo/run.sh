export SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct
export SLIME_SCRIPT_TRAIN_BACKEND=megatron
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM/:/root/sglang/python:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)  # or any free port


python examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn_grpo_npu.py
