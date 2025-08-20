export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


##############################
##############################

# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO
# export RCCL_MSCCLPP_THRESHOLD=1073741824
# export TORCH_BLAS_PREFER_HIPBLASLT=1
# export NCCL_MIN_NCHANNELS=112
# export RAY_CGRAPH_get_timeout=6000
# export NCCL_P2P_DISABLE=0
# export NCCL_P2P=1
# unset ROCR_VISIBLE_DEVICES
# export HYDRA_FULL_ERROR=1
# export FSDP_VERBOSE=1
# export TORCH_NCCL_HIGH_PRIORITY=1
# export NCCL_IB_GID_INDEX=3
# export RCCL_MSCCL_ENABLE=0
# export GPU_MAX_HW_QUEUES=2
# export NCCL_SOCKET_IFNAME=eth0
# export GLOO_SOCKET_IFNAME=eth0
# export NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7

##############################
##############################




# AMD特定设置
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=2

# 强制使用spawn方法
# export PYTORCH_MULTIPROCESSING_START_METHOD=spawn
# export USE_ROCM_AITER_ROPE_BACKEND=0
# export OMP_NUM_THREADS=1


### Debug Setting
# export NCCL_DEBUG=INFO #MUST to have ### on
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export MEGATRON_LOGGING_LEVEL=DEBUG
# export DIST_CHECKPOINTING_DEBUG=1

export DEBUG=3 #0~3
# export VERBOSE=1
# export LOG_LEVEL=DEBUG
# export LOGLEVEL=DEBUG
export LOGGING_LEVEL=DEBUG
export PYTHONUNBUFFERED=1  # 确保立即输出日志
# export TORCH_LOGS="+all"


#################
#################
# Multi-process
# export PYTORCH_MULTIPROCESSING_START_METHOD=spawn
# export OMP_NUM_THREADS=1

# Disable async altogether for single GPU
# export MEGATRON_DISABLE_ASYNC_SAVE=1


export PYTHONPATH=/home/yushensu/projects/slime:/workspace/Megatron-LM-amd_version


# export USE_ROCM_AITER_ROPE_BACKEND=0





# source scripts/models/qwen3-4B.sh
# PYTHONPATH=/home/yushensu/projects/slime:/workspace/Megatron-LM-amd_version python tools/convert_hf_to_torch_dist.py \
#     ${MODEL_ARGS[@]} \
#     --hf-checkpoint ../model/Qwen3-4B \
#     --save ../model/Qwen3-4B_torch_dist_amd_new \
#     --no-gradient-accumulation-fusion \
#     --ckpt-format torch_dist


# PYTHONPATH=/home/yushensu/projects/slime:/workspace/Megatron-LM-amd_version torchrun --nproc_per_node=8 mbridge/example/2.load_model_and_export_multiple_gpus.py \
#     --model_path ../model/Qwen3-4B \
#     --save_path ../model/Qwen3-4B_torch_dist_amd_new \
#     # --no-gradient-accumulation-fusion


################################
################################

# source scripts/models/qwen3-4B.sh
# PYTHONPATH=/home/yushensu/projects/slime:/workspace/Megatron-LM-amd_version python tools/convert_hf_to_torch_dist.py \
#     ${MODEL_ARGS[@]} \
#     --hf-checkpoint ../model/Qwen3-4B \
#     --save ../model/Qwen3-4B_torch_dist_amd_new \
#     --no-gradient-accumulation-fusion \
#     --async-save 


source scripts/models/qwen3-4B.sh
# torchrun --nproc_per_node=8 tools/convert_hf_to_torch_dist.py \
python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ../model/Qwen3-4B \
    --save ../model/Qwen3-4B_torch_dist_amd_new \
    --no-gradient-accumulation-fusion \
    # --async-save false
    # --ckpt-format torch_dist \
    # --async-save 


# source scripts/models/qwen3-4B.sh
# PYTHONPATH=/home/yushensu/projects/slime:/usr/local/lib/python3.12/dist-packages/core python tools/convert_hf_to_torch_dist.py \
#     ${MODEL_ARGS[@]} \
#     --hf-checkpoint ../model/Qwen3-4B \
#     --save ../model/Qwen3-4B_torch_dist_amd_new \
#     --no-gradient-accumulation-fusion \
#     --async-save 


# source scripts/models/qwen3-8B.sh
# PYTHONPATH=/home/yushensu/projects/slime:/workspace/Megatron-LM-amd_version python tools/convert_hf_to_torch_dist.py \
#     ${MODEL_ARGS[@]} \
#     --hf-checkpoint ../model/Qwen3-8B \
#     --save ../model/Qwen3-8B_torch_dist_amd_new \
#     --no-gradient-accumulation-fusion \
#     --async-save 