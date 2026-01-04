"""
Example to demonstrate how to launch training.
You can also do the same using a .sh or others, and we use Python here just for simplicity.
"""

import json
import os
import subprocess
from pathlib import Path

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[3]

MODEL_NAME, MODEL_TYPE = "Qwen3-8B", "qwen3-8B"

NUM_GPUS = 8

ckpt_args = (
    f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
    f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    "--save-interval 20 "
    f"--load /root/{MODEL_NAME}_slime "
    f"--save /root/{MODEL_NAME}_slime "
)

rollout_args = (
    "--prompt-data /root/datasets/formal_math_single_round/minimal_demo/flc_train.jsonl "
    "--input-key prompt "
    "--apply-chat-template "
    "--rollout-shuffle "
    "--custom-rm-path examples.formal_math.single_round.reward_fn.reward_fn "
    "--reward-key reward_value "
    "--log-reward-category reward_cat "
    "--rollout-batch-size 32 "
    "--n-samples-per-prompt 8 "
    "--rollout-max-response-len 8192 "
    "--rollout-temperature 1 "
    "--global-batch-size 256 "
    "--balance-data "
    "--num-rollout 3000 "
)

eval_args = (
    "--eval-interval 20 "
    "--n-samples-per-eval-prompt 1 "
    "--eval-max-response-len 16384 "
    "--eval-top-p 1 "
    "--eval-prompt-data "
    "minif2f /root/datasets/formal_math_single_round/minimal_demo/minif2f_test.jsonl "
)

perf_args = (
    "--tensor-model-parallel-size 2 "
    "--sequence-parallel "
    "--pipeline-model-parallel-size 1 "
    "--context-parallel-size 1 "
    "--expert-model-parallel-size 1 "
    "--expert-tensor-parallel-size 1 "
    "--recompute-granularity full "
    "--recompute-method uniform "
    "--recompute-num-layers 1 "
    "--use-dynamic-batch-size "
    "--max-tokens-per-gpu 6144 "
)

grpo_args = (
    "--advantage-estimator grpo "
    "--use-kl-loss "
    "--kl-loss-coef 0.00 "
    "--kl-loss-type low_var_kl "
    "--entropy-coef 0.00 "
    "--eps-clip 0.2 "
    "--eps-clip-high 0.28 "
)

optimizer_args = (
    "--optimizer adam "
    "--lr 1e-6 "
    "--lr-decay-style constant "
    "--weight-decay 0.1 "
    "--adam-beta1 0.9 "
    "--adam-beta2 0.98 "
)

sglang_args = "--rollout-num-gpus-per-engine 8 " "--sglang-mem-fraction-static 0.7 "

misc_args = (
    "--attention-dropout 0.0 "
    "--hidden-dropout 0.0 "
    "--accumulate-allreduce-grads-in-fp32 "
    "--attention-softmax-in-fp32 "
    "--attention-backend flash "
    "--actor-num-nodes 1 "
    "--actor-num-gpus-per-node 8 "
    "--colocate "
    "--log-passrate "
)

wandb_args = (
    "--use-wandb "
    "--wandb-project slime-formal-math-run-minimal "
    "--wandb-group demo "
    "--wandb-key ${WANDB_API_KEY} "
)

train_args = (
    f"{ckpt_args} "
    f"{rollout_args} "
    f"{optimizer_args} "
    f"{grpo_args} "
    f"{wandb_args} "
    f"{perf_args} "
    f"{eval_args} "
    f"{sglang_args} "
    f"{misc_args} "
)

runtime_env_json = json.dumps(
    {
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    }
)

cmd = (
    f"export PYTHONBUFFERED=16 && "
    f'source "{repo_base_dir}/scripts/models/{MODEL_TYPE}.sh" && '
    f'ray job submit --address="http://127.0.0.1:8265" '
    f"--runtime-env-json='{runtime_env_json}' "
    f"-- python3 train.py "
    "${MODEL_ARGS[@]} "
    f"{train_args}"
)

print(f"EXEC: {cmd}", flush=True)
subprocess.run(["bash", "-c", cmd], shell=False)
