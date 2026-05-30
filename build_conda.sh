#!/bin/bash

set -ex

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
export SGLANG_COMMIT="5a15cde858ea09b77116212a39356f2fc51b8584"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"

export BASE_DIR=${BASE_DIR:-"/root"}
cd $BASE_DIR

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

pip install cuda-python==12.9

# install sglang
if [ ! -d "$BASE_DIR/sglang" ]; then
  cd $BASE_DIR
  git clone https://github.com/sgl-project/sglang.git
fi
cd $BASE_DIR/sglang
git checkout ${SGLANG_COMMIT}
# Install sglang's Python package. Sglang pins `torch==2.11.0` and ships native
# kernels (sglang-kernel, sgl-deep-gemm) built against cu13 on pypi; pip's
# resolver insists on those during this editable install no matter what we
# pre-stage. Let it complete with cu13, then force-reinstall torch and the
# native kernels from the cu129 indexes so the env ends up consistently cu129.
pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url https://docs.sglang.ai/whl/cu129/
# sglang's editable install pulled in pypi's cu13 torch+kernels, which dragged
# along a full set of nvidia-*-cu13 / cuda-toolkit==13 wheels. The force-
# reinstalls above swap torch/sglang-kernel/sgl-deep-gemm to their +cu129
# variants but DO NOT remove those stale cu13 nvidia runtime libs, and they can
# get dlopen'd ahead of our cu12 libs at import time (e.g. deep_gemm's
# libcudart.so.13 lookup). Uninstall them explicitly so the env is purely cu12.
pip uninstall -y \
  cuda-toolkit \
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


pip install cmake ninja

# flash attn
# the newest version megatron supports is v2.7.4.post1
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.1
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 --no-cache-dir --force-reinstall
pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall

# megatron
cd $BASE_DIR
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd $BASE_DIR/Megatron-LM && git checkout ${MEGATRON_COMMIT} && pip install -e .

# install slime and apply patches

# if slime does not exist locally, clone it
if [ ! -d "$BASE_DIR/slime" ]; then
  cd $BASE_DIR
  git clone  https://github.com/THUDM/slime.git
  cd slime/
  export SLIME_DIR=$BASE_DIR/slime
  pip install -e .
else
  export SLIME_DIR=$BASE_DIR/slime
  cd $SLIME_DIR
  pip install -e .
fi

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"
# kernels 0.15.x trips a ValueError("Either a revision or a version must be
# specified") on `transformers.integrations.hub_kernels` import; pin to <0.15
# so `import sglang` works at runtime.
pip install "kernels<0.15.0"

# apply patch
cd $BASE_DIR/sglang
git apply --check $SLIME_DIR/docker/patch/v0.5.12.post1/sglang.patch 2>/dev/null && \
  git apply $SLIME_DIR/docker/patch/v0.5.12.post1/sglang.patch || \
  echo "sglang patch already applied or not applicable, skipping"
cd $BASE_DIR/Megatron-LM
git apply --check $SLIME_DIR/docker/patch/v0.5.12.post1/megatron.patch 2>/dev/null && \
  git apply $SLIME_DIR/docker/patch/v0.5.12.post1/megatron.patch || \
  echo "megatron patch already applied or not applicable, skipping"
