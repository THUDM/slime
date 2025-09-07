#!/bin/bash

set -ex

# create conda
yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc

micromamba create -n slime python=3.12 pip -c conda-forge -y
micromamba activate slime
export CUDA_HOME="$CONDA_PREFIX"

export BASE_DIR=${BASE_DIR:-"/root"}
cd $BASE_DIR
# install sglang
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout 8ecf6b9d2480c3f600826c7d8fef6a16ed603c3f
# Install the python packages
pip install -e "python[all]"

# install cuda 12.8 as it's the default cuda version for torch
micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev -c nvidia/label/cuda-12.8.0 -y
micromamba install -n slime -c conda-forge cudnn -y
pip install cmake ninja

# reinstall sglang deps
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git --no-cache-dir --force-reinstall --no-build-isolation

# install megatron deps
TORCH_CUDA_ARCH_LIST="9.0;9.0a" \
  pip -v install --no-build-isolation \
  git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
# apex
TORCH_CUDA_ARCH_LIST="9.0;9.0a" NVCC_APPEND_FLAGS="--threads 4" \
\
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git
# transformer engine
TORCH_CUDA_ARCH_LIST="9.0;9.0a" \
  pip -v install transformer_engine[pytorch]
# flash attn
# the newest version megatron supports is v2.7.4.post1
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1
# megatron
cd $BASE_DIR
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/
git checkout 48406695c4efcf1026a7ed70bb390793918dd97b
pip install -e .

# mbridge
pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps

# install slime and apply patches

# if slime does not exist locally, clone it
if [ ! -d "$BASE_DIR/slime" ]; then
  cd $BASE_DIR
  git clone  https://github.com/THUDM/slime.git
  cd slime/
  export SLIME_DIR=$BASE_DIR/slime
  pip install -e .
else
  export SLIME_DIR=$BASE_DIR/
  pip install -e .
fi


# apply patch
cd $BASE_DIR/sglang
git apply $SLIME_DIR/docker/patch/v0.5.0rc0-cu126/sglang.patch
cd $BASE_DIR/Megatron-LM
git apply $SLIME_DIR/docker/patch/v0.5.0rc0-cu126/megatron.patch
