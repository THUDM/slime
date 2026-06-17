# GLM-5.2 744B-A40B with 256xH100

This is the recommended 32-node, 256-H100 training example for [GLM-5.2](https://z.ai/blog/glm-5.2).

The recipe uses the GLM-5.2 BF16 checkpoint for Megatron training and the FP8 checkpoint for SGLang rollout. It assumes two Hugging Face repositories will be available:

- BF16: `zai-org/GLM-5.2`
- FP8: `zai-org/GLM-5.2-FP8`

## Environment Setup

For environment setup and dataset download, see [Example: Qwen3-4B](qwen3-4B.md). For multi-node training, make sure every node can access the same `$BASE_DIR` path.

### Download Model

```bash
hf download zai-org/GLM-5.2 --local-dir $BASE_DIR/GLM-5.2
hf download zai-org/GLM-5.2-FP8 --local-dir $BASE_DIR/GLM-5.2-FP8
```

The open-source GLM-5.2 config uses `model_type: glm_moe_dsa`, which slime maps onto
the DeepSeek-V3.2 bridge (`slime_plugins.mbridge.deepseek_v32`) since the two share the
same DSA weight layout.

### Convert Checkpoint

The training side needs the BF16 Hugging Face checkpoint converted to the Megatron torch_dist format. The torch_dist format is reshardable, so the conversion parallel layout does **not** need to match training; we use a layout that satisfies Megatron's expert-group constraint on the conversion node count.

Run the following on 4 nodes / 32 GPUs:

```bash
cd /root/slime
pip install -e . --no-deps
source scripts/models/glm5.2-744B-A40B.sh
PYTHONPATH=/root/Megatron-LM/ torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --tensor-model-parallel-size 8 \
   --pipeline-model-parallel-size 2 \
   --decoder-last-pipeline-num-layers 40 \
   --expert-model-parallel-size 16 \
   --expert-tensor-parallel-size 1 \
   --hf-checkpoint $BASE_DIR/GLM-5.2/ \
   --save $BASE_DIR/GLM-5.2_torch_dist/
```

Here, `MASTER_ADDR` is the IP of node0, and `NODE_RANK` is the current node index.

`MODEL_ARGS` includes `--allgather-cp`, a slime-only flag, so `tools/convert_hf_to_torch_dist.py` registers it too (it is a no-op for conversion). On 32 GPUs, Megatron requires `expert_tp(1) * expert_model_parallel * pp` to divide the world size, so we convert with `EP=16` (`1*16*2=32`). The resulting checkpoint still loads at training-time `EP=32` because torch_dist is reshardable.

## Run Training

From node0:

```bash
cd /root/slime
export BASE_DIR=/shared/path
export MASTER_ADDR=<node0-ip>
export HOSTFILE=$BASE_DIR/hostfile  # one worker IP per line, all 32 nodes
bash scripts/run-glm5.2-744B-A40B.sh
```

If `HOSTFILE` is not set, join the other nodes to the Ray cluster manually.

### Parameter Introduction

#### Model Configuration

`scripts/models/glm5.2-744B-A40B.sh` contains the GLM-5.2 DSA + cross-layer index sharing configuration: 256 routed experts, top-8 activation, 1 shared expert, and 78 layers total (3 dense + 75 MoE).

The DSA index sharing schedule, such as `index_topk_freq=4` and `index_skip_topk_offset=3`, is read from the Hugging Face config. The Megatron side uses the shared `slime_plugins.models.glm5.glm5:get_glm5_spec` provider and enables:

```bash
--allgather-cp
```

This makes DSA + context parallel use the allgather-CP layout, and the index-share provider gathers index K/V across the CP group.

#### Training Parallelism

The default script targets 32 nodes and 256 GPUs:

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 8
   --decoder-first-pipeline-num-layers 14
   --decoder-last-pipeline-num-layers 16
   --context-parallel-size 8
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1
   ...
)
```

`TP=4 * PP=8 * CP=8 = 256` GPUs form one training group (`DP=1`). The expert group constraint `expert_tp(1) * EP(32) * PP(8) = 256` divides the world size exactly (`expert_dp=1`).

DSA cross-layer index sharing requires every pipeline stage to **start** on a "computing" layer. With `index_topk_freq=4` / `index_skip_topk_offset=3`, the computing layers are 1, 2, 3, 7, 11, ..., 75. A uniform `78/8` split would start stages on skip layers and fail the index-share assertion in `get_glm5_spec`. We therefore use `--decoder-first-pipeline-num-layers 14` and `--decoder-last-pipeline-num-layers 16`, leaving 6 middle stages of `(78-14-16)/6 = 8` layers each. The stage starts land on global layers 1, 15, 23, 31, 39, 47, 55, 63 — all computing layers.

#### BF16 Training + FP8 Rollout

The launcher writes the default paths directly in `CKPT_ARGS` and `ROLLOUT_ARGS`, matching the style of the other example scripts:

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/GLM-5.2-FP8
   --ref-load $BASE_DIR/GLM-5.2_torch_dist
   --load $BASE_DIR/GLM-5.2_slime
   --save $BASE_DIR/GLM-5.2_slime
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   ...
)
```

`--hf-checkpoint` provides FP8 weights and the tokenizer for SGLang rollout; `--ref-load` is the Megatron torch_dist checkpoint converted from BF16. To debug BF16 rollout, change `--hf-checkpoint` in the script to `$BASE_DIR/GLM-5.2`.

#### SGLang Configuration

The rollout side runs with **prefill/decode (PD) disaggregation**: 1 prefill engine (64 GPU) + 3 decode engines (192 GPU) = 256 GPUs total (which must equal the colocated `rollout_num_gpus`). Each engine spans 64 GPUs with DP attention and `EP=64` (DeepEP's dispatch config map supports up to 160 EP ranks, so a single 256-GPU engine would be invalid). Prefill uses the `auto` DeepEP path; decode uses `low_latency` + `deep_gemm`. The split is configured via the `--sglang-config` YAML:

```yaml
sglang:
  - name: default
    server_groups:
      - worker_type: prefill
        num_gpus: 64
        num_gpus_per_engine: 64
        overrides: { deepep_mode: auto, ... }
      - worker_type: decode
        num_gpus: 192
        num_gpus_per_engine: 64
        overrides: { deepep_mode: low_latency, moe_runner_backend: deep_gemm, ... }
```

PD transfer runs over RDMA/IB with the mooncake backend:

```bash
--sglang-disaggregation-transfer-backend mooncake
--sglang-disaggregation-ib-device mlx5_100,...,mlx5_107
```

The rest of the rollout uses FP8 KV cache and the NSA + DeepEP backends:

```bash
SGLANG_ARGS=(
   --sglang-enable-dp-attention
   --sglang-ep-size 64
   --sglang-dp-size 64
   --sglang-kv-cache-dtype fp8_e4m3
   --sglang-nsa-decode-backend flashmla_kv
   --sglang-nsa-prefill-backend flashmla_sparse
   --sglang-attention-backend nsa
   ...
)
```

MTP / EAGLE speculative decoding is enabled using the model's own next-token-prediction layer (the GLM-5.2 checkpoint ships an MTP layer), so no separate draft model is needed:

```bash
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 4
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 5
```

`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` must cover the largest decode batch: `max cuda_graph_max_bs (decode group = 12) * speculative_num_draft_tokens (5) = 60`, rounded up to `64`. A value below this trips the DeepEP low-latency dispatch buffer assertion during the decode group's CUDA-graph capture.

#### Networking

DeepEP/NVSHMEM communication across nodes needs the IB-aware NCCL settings in the Ray runtime env (`NCCL_SOCKET_IFNAME`, `NCCL_IB_*`, `NCCL_NET_GDR_LEVEL`, `NCCL_P2P_LEVEL=NVL`, `NCCL_NVLS_ENABLE=0`, `MC_IB_PCI_RELAXED_ORDERING`, ...). The script defaults to `SOCKET_IFNAME=eth0`; set `SOCKET_IFNAME` before launch if your environment differs, and it will be written to `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `NCCL_SOCKET_IFNAME`. DeepEP also requires `NVSHMEM_DISABLE_NCCL=1`.
