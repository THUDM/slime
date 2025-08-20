#!/bin/bash


# bash scripts/run-qwen3-4B-amd.sh


####clear before training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python


set -euxo pipefail


### AMD Support ###
SLIME_DIR="/home/yushensu/projects/slime" # Need to change to your own path
export SLIME_DIR=$SLIME_DIR

MODEL_DIR="/home/yushensu/projects/model" # Need to change to your own path
export MODEL_DIR=$MODEL_DIR

DATA_DIR="/home/yushensu/projects/data"  # Need to change to your own path
export DATA_DIR=$DATA_DIR

# For AMD GPU
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"} # Must set to 1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"} #You can choose which gpus to use
####################


# ### AMD Support ### (If you do not istall, please install them)
# # # Clone and install Megatron-LMi-amd_version
# export MAX_JOBS=512
# cd $SLIME_DIR
# pip uninstall megatron-core -y
# if [ ! -d "Megatron-LM-amd_version" ]; then
#     git clone git@github.com:yushengsu-thu/Megatron-LM-amd_version.git
# else
#     echo "Megatron-LM-amd_version directory already exists, skipping clone"
# fi
# cd Megatron-LM-amd_version
# pip install -vvv -e . 
# cd $SLIME_DIR

# # Install slime
# pip install -e .
# ####################



# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Current Model convert script on AMD GPU has some issue, please download the converted model from here: https://huggingface.co/zyzshishui0627/models 

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-8B
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load ${MODEL_DIR}/Qwen3-8B_torch_dist
   # --ref-load ${MODEL_DIR}/Qwen3-8B_torch_dist_amd_new
   --load ${MODEL_DIR}/Qwen3-8B_slime/
   --save ${MODEL_DIR}/Qwen3-8B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

### AMD Support ###
# Need to fix some issue with torch_memory_saver in rocm to support larger  --sglang-mem-fraction-static 
# SGLANG_ARGS=(
#    --rollout-num-gpus-per-engine 2
#    --sglang-mem-fraction-static 0.7
# )
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.4
)
####################


MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   ### AMD Support ###
   # disable gradient accumulation fusion: Need to add apex to enable this
   --no-gradient-accumulation-fusion
   ###################
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=$(echo ${HIP_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# "PYTHONPATH": "/workspace/Megatron-LM-amd_version/",
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/workspace/Megatron-LM-amd_version/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}



####clear after training

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python










