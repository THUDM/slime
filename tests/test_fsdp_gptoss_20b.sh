pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# ref link in verl: https://github.com/volcengine/verl/pull/3212/files
cat > convert_model.py << EOF
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

model_id = "openai/gpt-oss-20b"
output_dir = "/root/models/gpt-oss-20b-bf16"

if os.path.exists(output_dir):
    print(f"Model already exists at {output_dir}, skipping conversion.")
else:
    print(f"Converting model from {model_id} to {output_dir}...")
    
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    # Patch config
    model.config.attn_implementation = "eager"
    
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    print("Conversion done.")
EOF

python3 convert_model.py


# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=4,5,6,7

CKPT_ARGS=(
   --hf-checkpoint /root/models/gpt-oss-20b-bf16
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 1000
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-response-len 2048
   --rollout-temperature 0.8

   --global-batch-size 16
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
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

SGLANG_ARGS=(
   # Set equal to the number of GPUs per node for colocated mode
   --rollout-num-gpus-per-engine 4
   --sglang-tensor-parallel-size 1
   --sglang-dtype bfloat16
   --sglang-decode-log-interval 1000
)


WANDB_ARGS=(
   --use-wandb
   --wandb-project "slime-fsdp-gpt"
   --wandb-group "20b-bf16"
   --wandb-key ${WANDB_API_KEY}
)

# launch the master node of ray in container
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   --train-backend fsdp \
   --bf16 \
   --attn-implementation eager \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
