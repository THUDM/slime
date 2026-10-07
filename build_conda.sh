#!/bin/bash

set -ex

export SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

# create conda
yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
export PS1=tmp
mkdir -p /root/.cargo/
touch /root/.cargo/env
source ~/.bashrc

# The micromamba installer writes `nodefaults` into ~/.condarc as a channel
# entry, which newer micromamba versions try to fetch as a real anaconda.org
# repo (it isn't — it's a meta-tag) and time out on. Strip it.
if [ -f ~/.condarc ]; then
  sed -i '/^\s*-\s*nodefaults\s*$/d' ~/.condarc
fi

micromamba create -n slime python=3.12 pip -c conda-forge -y
micromamba activate slime
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"

# Keep these in sync with docker/Dockerfile:
#   - SGLANG_IMAGE_TAG (ARG)            -> SGLANG_VERSION below
#   - MEGATRON_COMMIT (ARG)             -> MEGATRON_COMMIT below
#   - PATCH_VERSION (ARG)               -> PATCH_VERSION below
#   - TMS_COMMIT (ARG)                   -> TMS_COMMIT below
#   - FLASH_QLA_COMMIT (ARG)             -> FLASH_QLA_COMMIT below
export SGLANG_VERSION="v0.5.15.post1"
export SGLANG_COMMIT="0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
export MEGATRON_COMMIT="4b4acac9a1d28ea6829c8d4f566d75698a21249d"
export PATCH_VERSION="v0.5.15.post1"
export TMS_COMMIT="4d525cf378fdbfe7eb044909b8e66e112a508624"
export FLASH_QLA_COMMIT="821fd9d37ede18fdc2a4e707fefe3770bfc32e58"
export TRANSFORMER_ENGINE_VERSION="2.18.0"
export NCCL_VERSION="2.30.7"

export BASE_DIR=${BASE_DIR:-"/root"}
cd "$BASE_DIR"

# install cuda 12.9 as it's the default cuda version for torch
micromamba install -n slime \
  cuda=12.9.1 \
  cuda-nvtx=12.9.79 \
  cuda-nvtx-dev=12.9.79 \
  nccl \
  -c nvidia/label/cuda-12.9.1 \
  -c nvidia \
  -c conda-forge \
  -y
micromamba install -n slime -c conda-forge cudnn -y
# sglang's editable install builds a Rust extension (sglang-grpc via
# setuptools-rust), so the conda env needs a working rustc + cargo.
micromamba install -n slime -c conda-forge rust -y

# install sglang. The Dockerfile starts FROM slimerl/sglang:v0.5.15.post1-cu129
# which already has sglang installed with cu129-built native kernels; we have
# to install it ourselves here. Three follow-up steps clean up the cu13 spill:
#   1. restore cuda-python 12.9 after SGLang's >=13 dependency upgrades it;
#   2. force-reinstall torch / sglang-kernel / sgl-deep-gemm to their +cu129
#      wheels (pypi defaults are cu13);
#   3. uninstall the cu13 nvidia-* runtime libs sglang dragged in, then
#      reinstall the cu12 equivalents to repair the `site-packages/nvidia/*`
#      shared dirs (pip uninstall stomps libs co-owned across cu12/cu13).
if [ ! -d "$BASE_DIR/sglang" ]; then
  cd "$BASE_DIR"
  git clone https://github.com/sgl-project/sglang.git
fi
cd "$BASE_DIR/sglang"
git checkout ${SGLANG_COMMIT}
pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall "cuda-python==12.9"
pip install --force-reinstall --no-deps \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
  --index-url https://docs.sglang.ai/whl/cu129/
pip uninstall -y \
  nvidia-cublas \
  nvidia-cuda-cupti \
  nvidia-cuda-nvrtc \
  nvidia-cuda-runtime \
  nvidia-cudnn-cu13 \
  nvidia-cufft \
  nvidia-cufile \
  nvidia-curand \
  nvidia-cusolver \
  nvidia-cusparse \
  nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 \
  nvidia-nvjitlink \
  nvidia-nvshmem-cu13 \
  nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 \
  || true
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.16.0.29 \
  nvidia-cufft-cu12 \
  nvidia-cufile-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple

pip install cmake ninja

# Match Docker's FA2/FA3 wheels without compiling or resolving PyTorch again.
# FA2 uses the community build linked from upstream issue #2425; FA3 uses
# the official PyTorch CUDA 12 index. Both downloads use fixed versions.
pip uninstall -y flash-attn-4 flash_attn_4 || true
pip install --no-deps --force-reinstall \
  "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  "https://download.pytorch.org/whl/cu129/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"

pip install flash-linear-attention==0.5.2
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
pip install git+https://github.com/QwenLM/FlashQLA.git@${FLASH_QLA_COMMIT} --no-build-isolation
# tilelang (matches Dockerfile)
pip install tilelang==0.1.9 -f https://tile-ai.github.io/whl/nightly/cu128/

# Match Docker's official TE core wheel and compile only the PyTorch bindings.
# Use the pip NCCL library consistently instead of an older system copy.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/slime-nccl.sh" <<'SH'
export _SLIME_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export _SLIME_LD_LIBRARY_PATH_WAS_SET="${LD_LIBRARY_PATH+x}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
SH
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/slime-nccl.sh" <<'SH'
if [ -n "${_SLIME_LD_LIBRARY_PATH_WAS_SET:-}" ]; then
  export LD_LIBRARY_PATH="${_SLIME_OLD_LD_LIBRARY_PATH}"
else
  unset LD_LIBRARY_PATH
fi
unset _SLIME_OLD_LD_LIBRARY_PATH _SLIME_LD_LIBRARY_PATH_WAS_SET
SH
pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch
NVTE_WITH_NCCL_EP=0 MAX_JOBS=64 \
  pip install --no-build-isolation "transformer_engine[pytorch,core_cu12]==${TRANSFORMER_ENGINE_VERSION}"

NVCC_APPEND_FLAGS="--threads 1" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel ${MAX_JOBS}" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
# --no-build-isolation: TMS's setup.py needs to find nvcc + headers + the
# installed torch to build its cu${TMS_CUDA_MAJOR} native hook; pip's default
# PEP 517 build venv hides them, so the wheel comes out python-only (~46KB)
# and sglang trips `Only hook_mode=preload supports pauseable CUDA Graph`
# because the preload .so was never compiled in.
pip install -v git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} \
  --no-cache-dir --force-reinstall --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# megatron
cd "$BASE_DIR"
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
# Install Megatron's build dependencies for its editable build without isolation.
# The pybind11 extension provides megatron.core.datasets.helpers_cpp.
pip install "setuptools>=80.0.0" pybind11 "packaging>=24.2"
cd "$BASE_DIR/Megatron-LM" && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation

# Install runtime dependencies before reasserting the compatibility pins.

cd "$SLIME_DIR"
# Install slime's pure-python runtime deps first (wandb, ray, accelerate,
# transformers, etc.) from its requirements.txt, then install slime itself
# with --no-deps so pip doesn't re-resolve and stomp our pinned native libs
# (torch+cu129, sglang-kernel+cu129, ...). The Dockerfile does the same thing
# before the editable package installation.
pip install -r requirements.txt
pip install -e . --no-deps

# int4_qat kernel (matches Dockerfile)
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
pip install . --no-build-isolation

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy==1.26.4" "scipy==1.17.1"
# kernels 0.15.x trips a ValueError("Either a revision or a version must be
# specified") on `transformers.integrations.hub_kernels` import; pin to <0.15
# so `import sglang` works at runtime.
pip install "kernels<0.15.0"

# Resolving installs above can replace native libraries. Restore the CUDA 12
# wheels and the router fork after all dependency resolution has finished.
pip install "cuda-python==12.9"
pip install --force-reinstall --no-deps \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
  --index-url https://docs.sglang.ai/whl/cu129/
pip install --no-deps https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
pip install --no-deps --force-reinstall "nvidia-nccl-cu12==${NCCL_VERSION}"
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# Apply patches in the same order as Dockerfile.
patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
if [ ! -d "$patch_dir" ]; then
  echo "Patch directory does not exist: $patch_dir" >&2
  exit 1
fi

cd "$BASE_DIR/sglang"
for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch sglang-deterministic.patch; do
  patch_path="$patch_dir/${patch_name}"
  if [ ! -f "$patch_path" ]; then
    if [ "$patch_name" = "sglang.patch" ]; then
      echo "Required patch is missing: $patch_path" >&2
      exit 1
    fi
    continue
  fi
  if git apply --check "$patch_path"; then
    git apply "$patch_path"
  elif git apply --reverse --check "$patch_path"; then
    echo "$patch_name already applied, skipping"
  else
    echo "$patch_name does not apply cleanly" >&2
    exit 1
  fi
done
cd "$BASE_DIR/Megatron-LM"
for patch_name in megatron.patch megatron-sglang-aligned.patch; do
  patch_path="$patch_dir/${patch_name}"
  if [ ! -f "$patch_path" ]; then
    if [ "$patch_name" = "megatron.patch" ]; then
      echo "Megatron patch does not exist: $patch_path" >&2
      exit 1
    fi
    continue
  fi

  if git apply --reverse --check "$patch_path"; then
    echo "$patch_name already applied, skipping"
  else
    git update-index --refresh || true
    if ! git apply "$patch_path" --3way; then
      echo "$patch_name does not apply cleanly" >&2
      exit 1
    fi
    if git grep -n '^<<<<<<< ' -- .; then
      echo "$patch_name failed to apply cleanly. Please resolve conflicts." >&2
      exit 1
    fi
  fi
done

python - <<'PY'
from importlib.metadata import version
import ctypes

import sglang
import torch
import torchaudio
import torchvision
import transformer_engine.pytorch
from megatron.core import parallel_state

assert torch.__version__ == "2.11.0+cu129"
assert torchaudio.__version__ == "2.11.0+cu129"
assert torchvision.__version__ == "0.26.0+cu129"
assert torch.version.cuda == "12.9"
nccl_version = ctypes.c_int()
assert ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(nccl_version)) == 0
assert nccl_version.value == 23007
assert version("nvidia-nccl-cu12") == "2.30.7"
assert version("sglang") == "0.5.15.post1"
assert version("sglang-kernel").split("+")[0] == "0.4.4"
assert version("sgl-deep-gemm").split("+")[0] == "0.1.4"
assert version("cuda-python") == "12.9.0"
assert version("numpy") == "1.26.4"
assert version("scipy") == "1.17.1"
assert version("tilelang") == "0.1.9"
assert version("megatron-core").split("+")[0] == "0.19.2"
assert version("transformer-engine") == "2.18.0"
assert version("flash-attn").split("+")[0] == "2.8.3"
assert version("flash-attn-3") == "3.0.0"
assert version("flash-linear-attention") == "0.5.2"
assert version("fla-core") == "0.5.2"
assert hasattr(torch.ops.torchvision, "nms")
PY
