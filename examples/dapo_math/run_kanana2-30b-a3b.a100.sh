#!/bin/bash
#SBATCH --job-name=slime-kanana-2-30b-a3b-thinking-2601
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# NOTE: --container-* is NOT in the header so this script runs on the HOST,
#       giving access to srun / scontrol. Container options are passed per srun step.

set -ex

mkdir -p logs

# ── Cluster environment (SRUN_CONTAINER array requires source inside the script) ─
source "${REPO_ROOT}/exmaples/dapo_math/${SLIME_CLUSTER:-a100}.sh"

# ── NVLink detection (runs on host, same topology) ───────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


# ── Model args ───────────────────────────────────────────────────────────────
source "${REPO_ROOT}/examples/dapo_math/kanana-2-30b-a3b.sh"
# Note: UPSTREAM is set above for upstream code paths

# ── Hyperparameters (overridable via env) ────────────────────────────────────
LR=${LR:-1e-6}
LR_DECAY=${LR_DECAY:-constant}
GBS=${GBS:-1024}
ROLLOUT_BS=${ROLLOUT_BS:-128}
N_SAMPLES=${N_SAMPLES:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
MAX_RESP_LEN=${MAX_RESP_LEN:-31744}
EPS_CLIP=${EPS_CLIP:-0.2}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-0.28}
KL_COEF=${KL_COEF:-0.00}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/models/kakaocorp/kanana-2-30b-a3b-thinking-2601
   --ref-load $BASE_DIR/models/kakaocorp/kanana-2-30b-a3b-thinking-2601_torch_dist
   --load $BASE_DIR/models/kakaocorp/kanana-2-30b-a3b-thinking-2601_slime
   --save $BASE_DIR/models/kakaocorp/kanana-2-30b-a3b-thinking-2601_slime
   --save-interval $SAVE_INTERVAL
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/datasets/zhuzilin/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size $ROLLOUT_BS
   #--over-sampling-batch-size 256
   --n-samples-per-prompt $N_SAMPLES
   --rollout-max-response-len $MAX_RESP_LEN
   --rollout-temperature $TEMPERATURE

   --global-batch-size $GBS
   #--balance-data
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime24 $BASE_DIR/datasets/zhuzilin/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16000
   --eval-top-p 1
   --skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size $TP
   --sequence-parallel
   --pipeline-model-parallel-size $PP
   --context-parallel-size $CP
   --expert-model-parallel-size $EP
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu $MAX_TOKENS_PER_GPU
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef $KL_COEF
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip $EPS_CLIP
   --eps-clip-high $EPS_CLIP_HIGH
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr $LR
   --lr-decay-style $LR_DECAY
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-tensorboard
   --tb-project-name dev
   --tb-experiment-name kanana-2-30b-a3b-grpo-${ATTENTION_BACKEND}-${SLURM_JOB_NUM_NODES}n-tp${TP}pp${PP}cp${CP}ep${EP}-gbs${GBS}-n${N_SAMPLES}-temp${TEMPERATURE}-lr${LR}-mt${MAX_TOKENS_PER_GPU}-clip${EPS_CLIP}-kl${KL_COEF}-${SLIME_CLUSTER:-a100}
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group kanana-2-30b-a3b
   # --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.8
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-ep-size ${EP}
   --sglang-cuda-graph-bs 1 2 4 8 16 32 64 96 128
   --sglang-max-running-requests 512
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend ${ATTENTION_BACKEND}
   --sglang-attention-backend flashinfer
   --train-backend megatron
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
   --moe-permute-fusion
   --moe-router-fusion
   --cross-entropy-loss-fusion
   --cross-entropy-fusion-impl te
   --manual-gc
   --manual-gc-interval 10
)

# ── Node / GPU allocation ────────────────────────────────────────────────────
N_ROLLOUT_NODES=8                                          # nodes dedicated to rollout (sglang)
N_ACTOR_NODES=$((SLURM_JOB_NUM_NODES - N_ROLLOUT_NODES))  # remaining nodes for actor (training)
GPUS_PER_NODE=8
N_ROLLOUT_GPUS=$((N_ROLLOUT_NODES * GPUS_PER_NODE))
RAY_PORT=6379

# ── Ray multi-node setup ─────────────────────────────────────────────────────
# scontrol/srun work here because this script runs on the HOST (no --container-* in header)
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
export MASTER_ADDR="${HEAD_NODE_IP}"
export no_proxy="127.0.0.1,${HEAD_NODE_IP}"

echo "Ray head: ${HEAD_NODE} (${HEAD_NODE_IP}:${RAY_PORT})"

# ── Env vars that must be visible to every Ray worker (inherited via --export=ALL) ──
export PYTHONPATH="${UPSTREAM}:${MEGATRON_PATH}"
export NCCL_TIMEOUT_MS=36000000
export no_proxy="localhost,127.0.0.1,0.0.0.0,10.*,${HEAD_NODE_IP}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,10.*,${HEAD_NODE_IP}"

# ── Cleanup any leftover Ray processes on all nodes (no container needed) ────
# enroot runs in the host PID namespace, so host-level pkill reaches container procs
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
for NODE in $ALL_NODES; do
    srun --nodes=1 --ntasks=1 -w "$NODE" \
        bash -c "pkill -9 -u \$(whoami) -f 'gcs_server|raylet|plasma_store|dashboard' 2>/dev/null; true" &
done
wait
sleep 5

# Start Ray head (inside container)
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" "${SRUN_CONTAINER[@]}" \
    bash -c "source ${VENV}/bin/activate && ray start --head \
        --node-ip-address=${HEAD_NODE_IP} \
        --port=${RAY_PORT} \
        --num-gpus=${GPUS_PER_NODE} \
        --disable-usage-stats \
        --block" &

sleep 10

# Start Ray workers one-by-one with stagger delay (avoids pyxis container storms)
worker_num=$((SLURM_JOB_NUM_NODES - 1))
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i at ${node_i}"
    srun --nodes=1 --ntasks=1 -w "$node_i" "${SRUN_CONTAINER[@]}" \
        bash -c "source ${VENV}/bin/activate && ray start \
            --address=${HEAD_NODE_IP}:${RAY_PORT} \
            --num-gpus=${GPUS_PER_NODE} \
            --disable-usage-stats \
            --block" &
    sleep 1
done

sleep 10
echo "Ray cluster ready."

# ── Run training directly on head node (srun --overlap, no ray job submit) ────
TRAIN_SCRIPT="${REPO_ROOT}/logs/slime_train_${SLURM_JOB_ID}.sh"
cat > "$TRAIN_SCRIPT" << TRAIN_EOF
#!/bin/bash
source ${VENV}/bin/activate
python ${UPSTREAM}/train.py \
   --actor-num-nodes ${N_ACTOR_NODES} \
   --actor-num-gpus-per-node ${GPUS_PER_NODE} \
   --rollout-num-gpus ${N_ROLLOUT_GPUS} \
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
TRAIN_EOF

echo "Starting MAIN training process on ${HEAD_NODE}"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" "${SRUN_CONTAINER[@]}" \
    bash "$TRAIN_SCRIPT"

rm -f "$TRAIN_SCRIPT"

# Tear down Ray on all nodes (no container needed)
for NODE in $ALL_NODES; do
    srun --nodes=1 --ntasks=1 -w "$NODE" \
        bash -c "pkill -9 -u \$(whoami) -f 'gcs_server|raylet|plasma_store|dashboard' 2>/dev/null; true" &
done
wait
