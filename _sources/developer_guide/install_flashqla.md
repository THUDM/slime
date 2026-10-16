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

## Current CUDA 13 builds

SGLang v0.5.21 requires TileLang 0.1.12. The FlashQLA revision used by the
older CUDA 12 images requires TileLang 0.1.9, so the current Docker and conda
builds use the default FLA backend.

For the historical CUDA 12/FlashQLA environment, use a versioned CUDA 12 image
or build the previous checkout, whose Docker and conda scripts retain those
pins:

```bash
git worktree add ../slime-cu12 2f4280e3
cd ../slime-cu12
docker build -f docker/Dockerfile . -t slime:flashqla-cu12
```

## GB10 Image

`docker/Dockerfile.gb10` omits FlashQLA by default. To build it experimentally on GB10, use:

```bash
docker build -f docker/Dockerfile.gb10 . \
  --build-arg INSTALL_FLASHQLA=1 \
  -t slime:gb10-flashqla
```

Validate compilation and runtime behavior on that platform before using the backend in training.
