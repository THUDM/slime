# Setting up the Environment from Scratch

[中文版](../zh/build.md)

If it is inconvenient to directly use our pre-built image, we provide the following solution for setting up the environment:

## Setting up the environment based on anaconda / mamba

Here, we take micromamba as an example to build a conda environment named `slime` within the official sglang image `lmsysorg/sglang:latest`:

```bash
####################
# create conda
####################
yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
micromamba self-update

micromamba create -n slime python=3.10 pip -c conda-forge -y
# install cuda-12.6.0 as this is the default cuda version for pytorch
# and apex need this alignment.
micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev -c nvidia/label/cuda-12.6.0 -y
micromamba install -n slime -c conda-forge cudnn -y
micromamba run -n slime pip install cmake ninja

####################
# sglang deps
####################
export BASE_DIR=/root/
cd $BASE_DIR
git clone https://github.com/sgl-project/sglang.git --branch v0.4.9.post2 --depth 1
cd $BASE_DIR/sglang/
micromamba run -n slime pip -v install -e "python[all]"
# TODO: change to pip install sglang-router after it has a new release
micromamba run -n slime pip install sglang-router --force-reinstall

####################
# megatron deps
####################
TORCH_CUDA_ARCH_LIST="9.0;9.0a" micromamba run -n slime \
  pip -v install --no-build-isolation \
  git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
# apex
TORCH_CUDA_ARCH_LIST="9.0;9.0a" NVCC_APPEND_FLAGS="--threads 4" \
micromamba run -n slime \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git
# transformer engine
TORCH_CUDA_ARCH_LIST="9.0;9.0a" micromamba run -n slime \
  pip -v install transformer_engine[pytorch]
# flash attn
# the newest version megatron supports is v2.7.4.post1
micromamba run -n slime pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# megatron
cd /root/
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/
micromamba run -n slime pip install -e .

####################
# other deps
####################
micromamba run -n slime pip install git+https://github.com/zhuzilin/cumem_allocator.git --no-build-isolation
micromamba run -n slime pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps

####################
# slime
####################
cd $BASE_DIR
git clone https://github.com/THUDM/slime.git
cd slime/
micromamba run -n slime pip install -e .
# apply patch
cd $BASE_DIR/sglang
git apply /root/slime/docker/patch/sglang.patch
```
