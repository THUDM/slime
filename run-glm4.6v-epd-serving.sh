#!/bin/bash

# GLM-4.6V EPD (Encoder-Prefill-Decode) disaggregation test script
# 4 nodes (32 GPUs): E=8 GPUs (2 engines), P=8 GPUs (2 engines), D=16 GPUs (4 engines)
# All engines: TP=4, DP=4, dp_attention

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
export PYTHONBUFFERED=16


CKPT_ARGS=(
   --hf-checkpoint /cloud/oss_checkpoints/zai-org/GLM-4.6V
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/o1_alicloud/personal/zzl/rl_data/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1.0

   --global-batch-size 64
   --rollout-stop-token-ids 151329 151336 151338
)

EVAL_ARGS=(
   --eval-prompt-data aime /mnt/o1_alicloud/personal/zzl/rl_data/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 8
   --context-parallel-size 2
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --decoder-first-pipeline-num-layers 20

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   #--use-wandb
   --wandb-project slime-dev
   --wandb-group glm4.6v
)

SGLANG_CONFIG_FILE=$(mktemp /tmp/sglang_config_XXXXXX.yaml)
cat > "$SGLANG_CONFIG_FILE" <<'EOF'
sglang:
  - name: default
    server_groups:
      # Phase 1: encoder engines (standalone encode_server, no dp_attention)
      - worker_type: encoder
        num_gpus: 8
        num_gpus_per_engine: 4
        overrides:
          tp_size: 4
          disable_cuda_graph: true
          #encoder_only: true

      # Phase 2: prefill (language_only + encoder_urls auto-injected)
      # NOTE: dp_attention is NOT compatible with zmq_to_scheduler encoder
      #       transfer — only some DP schedulers register endpoints, causing
      #       the encoder to timeout waiting for all tp_size endpoints.
      - worker_type: prefill
        num_gpus: 8
        num_gpus_per_engine: 4
        overrides:
         tp_size: 4
         #language_only: true

      # Phase 2: decode (dp_attention is fine here, no encoder interaction)
      - worker_type: decode
        num_gpus: 16
        num_gpus_per_engine: 4
        overrides:
          dp_size: 4
          tp_size: 4
          enable_dp_attention: true
          enable_dp_lm_head: true
          #language_only: true

EOF
echo "sglang config written to $SGLANG_CONFIG_FILE"
cat "$SGLANG_CONFIG_FILE"


SGLANG_ARGS=(
   --rollout-function-path slime.rollout.sleep_rollout.sleep
   #--use-slime-router
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.78

   # --sglang-moe-a2a-backend deepep
   # --sglang-deepep-mode auto

   # hicache
   --sglang-enable-hierarchical-cache
   --sglang-hicache-size 80
   --sglang-hicache-write-policy write_back

   # dsa
   # --sglang-page-size 64
   # --sglang-nsa-decode-backend flashmla_sparse
   # --sglang-nsa-prefill-backend flashmla_sparse
   # --sglang-attention-backend nsa
   --sglang-cuda-graph-max-bs 8
   #--sglang-disable-cuda-graph
   --sglang-max-running-requests 512

   --sglang-config "$SGLANG_CONFIG_FILE"

   --sglang-watchdog-timeout 3600
   --router-disable-health-check

   --debug-rollout-only
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

   --moe-token-dispatcher-type flex
   #--moe-enable-deepep
)

# launch the master node of ray in container
export MASTER_ADDR=${MLP_WORKER_0_HOST}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats
sleep 5

for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
  if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
    continue
  fi
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
done
wait

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "MASTER_ADDR": "${MLP_WORKER_0_HOST}",
        "PYTHONPATH": "/root/Megatron-LM",
        "NCCL_CUMEM_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
        "NCCL_IB_TC": "160",
        "NCCL_PXN_DISABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_NET_GDR_LEVEL": "4",
        "NCCL_IB_RETRY_CNT": "7",
        "NCCL_IB_TIMEOUT": "32",
        "NCCL_IB_QPS_PER_CONNECTION": "8",
        "NCCL_P2P_LEVEL": "NVL",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_MIN_CTAS": "4",
        "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_btl": "^openib",
        "OMPI_MCA_routed": "direct",
        "OMPI_MCA_routed_radix": "1024",
        "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
        "OMPI_MCA_oob_tcp_if_include": "${MLP_SOCKET_IFNAME}",
        "OMPI_MCA_btl_tcp_if_include": "${MLP_SOCKET_IFNAME}",
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "40",
        "NVSHMEM_DISABLE_NCCL": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --save-debug-rollout-data "data.pt" \
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
