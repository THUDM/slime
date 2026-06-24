#!/bin/bash

# RL on the clear-obstacles grid game (real-time/environment/clear_obstacles).
# The model plays a grid game via move_up/move_down/move_left/move_right tool
# calls; reward is 1.0 for reaching the GOAL row, 0.0 otherwise.

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

# Help flashinfer's JIT compiler find libcuda: the conda toolchain is sandboxed and
# does not search system lib dirs, and the env's lib64/stubs path does not exist.
# Point the linker at the existing conda stub dirs so -lcuda resolves.
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib/stubs:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
export LDFLAGS="-L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs -L$CONDA_PREFIX/lib/stubs $LDFLAGS"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Repo root that contains slime/, real-time/, Megatron-LM/ (this script lives at
# slime/examples/realtime/), used to put the obstacles `environment` package on
# PYTHONPATH.
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
source "slime/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint $HOME/Qwen/Qwen3-4B
   --ref-load $HOME/Qwen/Qwen3-4B_torch_dist
   # --load $HOME/Qwen3-4B_slime/
   --save $HOME/Qwen/Qwen3-4B/qwen3-4b-obstacles/
   --save-interval 20
   --rotary-base 1000000
)

ROLLOUT_ARGS=(
   # Seed dataset produced by obstacles_data_preprocess.py. Each row is
   # {"prompt": <game rules>, "seed": <int>}; the seed arrives on sample.label
   # and the env is reconstructed from it at rollout time.
   --prompt-data $HOME/obstacles-seeds/train.jsonl
   --input-key prompt
   --label-key seed
   --rollout-shuffle
   --reward-key score
   --num-rollout 100
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   # Dump every rollout's samples (decoded completions + prompt/tokens/loss_mask/
   # metadata via Sample.to_dict) to a per-step .pt file for inspection during
   # training. {rollout_id} is filled in by slime, not bash.
   --save-debug-rollout-data $HOME/qwen3-4b-obstacles/rollout_dumps/{rollout_id}.pt

   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
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
   --max-tokens-per-gpu 17408
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
   --use-wandb
   --wandb-project slime-obstacles
   --wandb-group qwen3-4B-clear-obstacles
   #--wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_obstacles.generate
   --custom-rm-path generate_with_obstacles.reward_func
)

# Cap the open-file limit: the default (1048576) triggers a raylet SIGABRT crash
# ("Too many open files") in gRPC/boost-asio, which kills the dashboard job agent
# and makes `ray job submit` fail with a 500 / ServerDisconnectedError.
ulimit -n 65535

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution.
# ${REPO_ROOT}/real-time puts the obstacles `environment` package on PYTHONPATH
# for the ray rollout workers; ${SCRIPT_DIR} makes generate_with_obstacles importable.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"./Megatron-LM/:${SCRIPT_DIR}:./slime:${REPO_ROOT}/real-time\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"LIBRARY_PATH\": \"${LIBRARY_PATH}\",
    \"LDFLAGS\": \"${LDFLAGS}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
