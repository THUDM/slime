# Qwen3-Next-80B-A3B

## Environment and Data

Follow the [quick start](../get_started/quick_start.md) to choose a Docker image and download the DAPO training and AIME evaluation datasets. Use `build_conda.sh` for a CUDA 12.9 environment without Docker.

The release environment includes FLA for Gated Delta Net. It also includes the optional [FlashQLA backend](../developer_guide/install_flashqla.md) on CUDA 12; select it with `--qwen-gdn-backend flashqla` when adapting the launcher. CUDA 13 uses FLA.

Download the model to the same base directory as the datasets:

```bash
export BASE_FOLDER=/root
hf download Qwen/Qwen3-Next-80B-A3B-Thinking \
  --local-dir "${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking"
```

## Checkpoint Conversion

```bash
cd /root/slime
source scripts/models/qwen3-next-80B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking" \
  --save "${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking_torch_dist"
```

The model uses a custom Megatron layer specification. See [model architectures beyond Megatron](../advanced/arch-support-beyond-megatron.md) for how the GDN and attention layers integrate with training.

## Training

For one node with eight GPUs:

```bash
cd /root/slime
export BASE_FOLDER=/root
export MASTER_ADDR=127.0.0.1
ACTOR_NUM_NODES=1 CP_SIZE=1 bash scripts/run-qwen3-next-80B-A3B.sh
```

For four nodes with eight GPUs each, place the model and datasets at the same path on every node and prepare a hostfile:

```bash
cd /root/slime
export BASE_FOLDER=/root
export MASTER_ADDR=your_master_addr
export HOSTFILE=/path/to/hostfile
bash scripts/run-qwen3-next-80B-A3B.sh
```

The launcher's defaults are `ACTOR_NUM_NODES=4`, `ACTOR_NUM_GPUS_PER_NODE=8`, and `CP_SIZE=4`. Set them explicitly when changing the node count or parallelism. The H100/H200 recipe is the reference setup; validate the selected kernels and parallel configuration when adapting it to Blackwell.

If gradient accumulation consumes too much memory, replace `--accumulate-allreduce-grads-in-fp32` with `--grad-reduce-in-bf16` in the launch script.
