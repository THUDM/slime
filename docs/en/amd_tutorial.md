# AMD Tutorial

⚠️ If you encounter problems on AMD instinct, feel free to reach out [Yusheng Su](https://yushengsu-thu.github.io/). 


## Introduction

If you are running Slime on AMD's Instinct, please refer to the following materials. This tutorial will explain how to set up the development environment (Docker), use the modified ROCm dependencies, and provide an example for running the experiments. The current rocm docker only support AMD's MI300 and MI325 GPUs.


<!-- First, you need to configure the slime runtime environment according to the [Readme](../../README.md) documentation and cd to the slime project directory. -->

## Docker

You can download the prebuilt image from DockerHub: [rlsys/slime](https://hub.docker.com/r/rlsys/slime/tags). 
```bash
docker pull rlsys/slime:slime_ubuntu22.04_rocm6.3.4-patch-numa-patch_sglang0.4.9_megatron-core-patch_ray2.47.1_apex_torch-memory-saver0.0.8-patch
```
Or you can use the [Dockerfile.rocm](docker/Dockerfile.rocm) to build it on your side.
```bash
cd docker
docker build -f Dockerfile.rocm -t slime_ubuntu22.04_rocm6.3.4-patch-numa-patch_sglang0.4.9_megatron-core-patch_ray2.47.1_apex_torch-memory-saver0.0.8-patch
```

Acknowledgement: Thanks to [Yang Wang](https://www.microsoft.com/en-us/research/people/yangwang5/) for working on the patch for this ROCm base Docker image to support virtual memory management on MI300X.


## Quick Start

### Environment Setup

Based on the [rlsys/slime](https://hub.docker.com/r/rlsys/slime/tags) image (pre-installed with SGLang 0.4.9 and Megatron-LM):
```bash
docker run --rm -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  --name slime_dev \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -w $PWD \
  rlsys/slime:slime_ubuntu22.04_rocm6.3.4-patch-numa-patch_sglang0.4.9_megatron-core-patch_ray2.47.1_apex_torch-memory-saver0.0.8-patch \
  /bin/bash
```

Then, download and install slime.
```bash
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
```


### Checkpoint Format Conversion

Since slime uses Megatron, and Megatron does not support loading Hugging Face checkpoints directly, we need to convert the model to the `torch_dist` format that Megatron supports.

#### HF → Megatron torch\_dist ckpt

Use [mbridge](https://github.com/ISEEKYAN/mbridge.git) or [Megatron-LM-amd_version-amd](https://github.com/yushengsu-thu/Megatron-LM-amd_version.git) for conversion:

```bash
cd slime/
# PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')
PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint model/Qwen3-4B \
    --save model/Qwen3-4B_torch_dist
```

⚠️ If you encounter an issue where slime cannot be found, please run `pip install -e .` in the slime directory.


### Example: Qwen3-4B

We provide examples to use [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), please refer to:
- [Example: Qwen3-4B Model](scripts/run-qwen3-4B-amd.sh): Just run `scripts/run-qwen3-4B-amd.sh` 

⚠️ TODO: The [ROCm-version torch_memory_saver](https://github.com/yushengsu-thu/torch_memory_saver.git) does not seem to clear memory properly; thus, we set `--sglang-mem-fraction-static` as `0.4` currently. We will continue investigating and focus on ROCm's virtual memory management for further modifications.

⚠️ TODO: ROCM seems to not support `apex` yet. Thus, we need to disable `--no-gradient-accumulation-fusion` currently. We will continue investigating how to enable this. 

⚠️ Note: The main difference between ROCm's training script and NVIDIA's script is that you need to set `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` for ray to function properly on AMD GPUs.

- We show the training script below: 

```bash
#!/bin/bash

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

### ROCm Support ###
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

# ### ROCm Support ### (If you do not istall, please install them)
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


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load ${MODEL_DIR}/Qwen3-4B_torch
   --load ${MODEL_DIR}/Qwen3-4B_slime/
   --save ${MODEL_DIR}/Qwen3-4B_slime/
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
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats


# "PYTHONPATH": "$(dirname $(python3 -c 'import megatron.core; print(megatron.core.__file__)'))"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "'${SLIME_DIR}'/Megatron-LM-amd_version/",
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
```
