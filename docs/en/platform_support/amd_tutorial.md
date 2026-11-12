# AMD ROCm

This tutorial explains how to run slime with the ROCm image.

Supported GPU architectures:

- MI300X / MI325X (gfx942)
- MI350X / MI355X (gfx950)

This tutorial uses gfx942 as the example; the workflow for gfx950 is the same. To use gfx950, set `--build-arg GPU_TARGET` to `gfx950` when building the image, and replace the `_gfx942` suffix of the image tag in the following commands with `_gfx950`.

## Environment Setup

Build the image (run from the root of the slime repository):

```bash
docker build --build-arg GPU_TARGET=gfx942 -t amdagi/slime:rocm7.15_torch2.12_py3.12_gfx942 -f docker/Dockerfile.rocm .
```

Or pull the prebuilt image:

```bash
docker pull amdagi/slime:rocm7.15_torch2.12_py3.12_gfx942
```

Start the container:

```bash
docker run --rm -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  amdagi/slime:rocm7.15_torch2.12_py3.12_gfx942 /bin/bash
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

Before running weight conversion or training, run the following command:

```bash
cat > /root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py <<'EOF'
import importlib
import sys

sys.modules[__name__] = importlib.import_module("nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async")
EOF
```

Otherwise both weight conversion and training fail with:

```text
ModuleNotFoundError: No module named 'megatron.core.dist_checkpointing.strategies.filesystem_async'
```

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

Replace `PYTHONPATH` with the actual Megatron-LM install path in the image.

## Training

```bash
cd /root/slime
SLIME_DIR=/root MODEL_DIR=/root DATA_DIR=/root \
  bash scripts/run-qwen3-4B-amd.sh
```

The script disables gradient-accumulation fusion with `--no-gradient-accumulation-fusion`. Ray GPU visibility is controlled by `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`; keep these settings when modifying the launch script.
