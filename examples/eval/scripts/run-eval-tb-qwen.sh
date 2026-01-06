#!/bin/bash

# Example launcher that reuses the Qwen3-8B recipe but delegates evaluation to an
# external Terminal Bench server via the eval_delegate_rollout wrapper.

# Clean up any stale processes from a previous run.
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16
export SLIME_HOST_IP=${SLIME_HOST_IP:-"127.0.0.1"}

MODEL_DIR="${MODEL_DIR:-/root/.cache}"
export MODEL_DIR

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3-8B.sh"

# Store eval/delegate settings in a YAML config similar to examples/eval_multi_task.
EVAL_CONFIG_PATH=${TB_EVAL_CONFIG_PATH:-"${REPO_ROOT}/examples/eval/scripts/eval_tb_example.yaml"}

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/OpenThinker-Agent-v1 # huggingface-cli download open-thoughts/OpenThinker-Agent-v1
   --ref-load ${MODEL_DIR}/OpenThinker-Agent-v1_torch_dist
   # --load ${MODEL_DIR}/OpenThinker-Agent-v1_slime/
   --save ${MODEL_DIR}/OpenThinker-Agent-v1_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
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
   --eval-interval 5
   --eval-config "${EVAL_CONFIG_PATH}"
   --eval-function-path examples.eval.eval_delegate_rollout.generate_rollout
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

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
   --use-wandb
   --wandb-project slime-eval
   --wandb-group qwen3-8b-eval
   --wandb-key ${WANDB_KEY}   # export WANDB_KEY="your_key"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-router-port 30005
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_VISIBLE_DEVICES=0,1

ray start --head --node-ip-address ${MASTER_ADDR} --port 6380 --num-gpus 2 \
            --disable-usage-stats \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=8266 \
            --dashboard-agent-listen-port 52366 \
            --dashboard-agent-grpc-port 52367 \
            --runtime-env-agent-port 52368


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://${MASTER_ADDR}:8266" \
   --working-dir "${REPO_ROOT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
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
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
