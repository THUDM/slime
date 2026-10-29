# AMD ROCm

This tutorial uses the ROCm image for MI300/MI325.

## Environment Setup

Build the image and start a container:

```bash
cd docker
docker build -f Dockerfile.rocm -t rlsys/slime:latest .
docker run --rm -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  rlsys/slime:latest /bin/bash
```

## Model and Data

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

## Weight Conversion

On ROCm, the shared conversion script requires `--use-cpu-initialization` so that model weights are initialized and saved on CPU.

```bash
cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/workspace/Megatron-LM-amd_version python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion \
  --use-cpu-initialization \
  --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
  --save /root/Qwen3-4B-Instruct-2507_torch_dist
```

Replace `PYTHONPATH` with the Megatron-LM install path in the image.

## Training

```bash
cd /root/slime
SLIME_DIR=/root MODEL_DIR=/root DATA_DIR=/root \
  bash scripts/run-qwen3-4B-amd.sh
```

This recipe disables gradient-accumulation fusion with `--no-gradient-accumulation-fusion`. Ray GPU visibility is controlled by `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`; keep these settings when adapting the launch script.
