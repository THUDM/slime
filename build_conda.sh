#!/bin/bash

set -ex

# create conda
yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
export PS1=tmp
mkdir -p /root/.cargo/
touch /root/.cargo/env
source ~/.bashrc

micromamba create -n slime python=3.12 pip -c conda-forge -y
micromamba activate slime
export CUDA_HOME="$CONDA_PREFIX"

export BASE_DIR=${BASE_DIR:-"/root"}
cd $BASE_DIR

# install cuda 12.9 as it's the default cuda version for torch
micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
micromamba install -n slime -c conda-forge cudnn -y

# prevent installing cuda 13.0 for sglang
pip install cuda-python==13.1.0
TORCH_VERSION=${TORCH_VERSION:-"2.4.1"}
TORCH_MINOR=${TORCH_VERSION%.*}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-"0.19.1"}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-"2.4.1"}
TORCH_CUDA_INDEX=${TORCH_CUDA_INDEX:-"https://download.pytorch.org/whl/cu124"}
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url ${TORCH_CUDA_INDEX}

pip install cmake ninja

# install sglang
SGLANG_COMMIT=5e2cda6158e670e64b926a9985d65826c537ac82
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout ${SGLANG_COMMIT}
# Install the python packages
pip install -e "python[all]"

# flash attn
# Download prebuilt wheel from the official release to avoid compilation.
# The default torch pin (2.4.1) matches the prebuilt v2.7.4.post1 wheel tested on 4090; override TORCH_VERSION to use another compatible release.
FLASH_ATTN_VERSION=2.7.4.post1
FLASH_ATTN_BASE="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}"
FLASH_TORCH_TAG="torch${TORCH_MINOR}"
FLASH_ATTN_WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu12${FLASH_TORCH_TAG}cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
pip install "${FLASH_ATTN_BASE}/${FLASH_ATTN_WHEEL}"

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --prefer-binary --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.0
pip install --prefer-binary --no-build-isolation nvidia-apex

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# megatron
cd $BASE_DIR
MEGATRON_COMMIT=core_v0.14.0
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
  cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
  pip install -e .

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29

# install slime and apply patches

SLIME_DIR=${SLIME_DIR:-"$BASE_DIR/slime"}
if [ ! -d "$SLIME_DIR" ]; then
  cd $BASE_DIR
  git clone https://github.com/THUDM/slime.git
fi
cd $SLIME_DIR
pip install -e .

# apply patch
cd $BASE_DIR/sglang
git apply $SLIME_DIR/docker/patch/v0.5.6/sglang.patch
cd $BASE_DIR/Megatron-LM
git apply $SLIME_DIR/docker/patch/v0.5.6/megatron.patch