# Installing FlashQLA

FlashQLA is an optional Gated Delta Net (GDN) backend for Qwen3-Next and Qwen3.5. To select it, add this argument to the training command:

```bash
--qwen-gdn-backend flashqla
```

The default backend is FLA.

## Requirements

- PyTorch 2.8 or newer.
- CUDA 12.8 or newer.
- NVIDIA SM90 or newer GPUs.
- The same FlashQLA installation on every training node.

## Conda and CUDA 12 Docker Images

`build_conda.sh` and the standard CUDA 12 Docker image install FlashQLA with TileLang 0.1.9. For a local environment, run:

```bash
bash build_conda.sh
```

To build the CUDA 12 image:

```bash
docker build -f docker/Dockerfile . \
  --build-arg SGLANG_IMAGE_TAG=v0.5.15.post1-cu129 \
  -t slime:flashqla
```

The CUDA 13 image uses TileLang 0.1.11 for SGLang, which is incompatible with the current FlashQLA requirement. That image uses the default FLA backend.

## GB10 Image

`docker/Dockerfile.gb10` omits FlashQLA by default. To build it experimentally on GB10, use:

```bash
docker build -f docker/Dockerfile.gb10 . \
  --build-arg INSTALL_FLASHQLA=1 \
  -t slime:gb10-flashqla
```

Validate compilation and runtime behavior on that platform before using the backend in training.
