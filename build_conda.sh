#!/bin/bash

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA_BIN="${HOME}/.local/bin"

# install micromamba if not present
if ! command -v micromamba &>/dev/null; then
  BIN_FOLDER="$MAMBA_BIN" INIT_YES=no CONDA_FORGE_YES=no \
    bash "$SCRIPT_DIR/.downloads/mamba_install.sh" < /dev/null
fi
export PATH="$MAMBA_BIN:$PATH"
eval "$(micromamba shell hook -s bash)"

MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

# If slime env was already restored by custom_proxy.sh, skip conda setup
if [ -d "$MAMBA_ROOT/envs/slime" ]; then
  echo "Conda env 'slime' already exists at $MAMBA_ROOT/envs/slime, skipping conda setup."
  micromamba activate slime
else
  # Use --offline if package cache exists
  if [ -d "$MAMBA_ROOT/pkgs" ] && [ "$(ls -A "$MAMBA_ROOT/pkgs" 2>/dev/null)" ]; then
    OFFLINE_FLAG="--offline"
  else
    OFFLINE_FLAG=""
  fi
  micromamba create -n slime python=3.12 pip -c conda-forge $OFFLINE_FLAG -y
  micromamba activate slime
  # install cuda 12.9 as it's the default cuda version for torch
  micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev nccl \
    -c nvidia/label/cuda-12.9.1 -c conda-forge $OFFLINE_FLAG -y
  micromamba install -n slime cudnn -c conda-forge $OFFLINE_FLAG -y
fi

export CUDA_HOME="$CONDA_PREFIX"
export SGLANG_COMMIT="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"
export DEEPEP_COMMIT="${DEEPEP_COMMIT:-main}"

# Patch set selector. Supported:
#   v0.5.9       - upstream default (sglang + megatron patches only)
#   v0.5.9.a100  - A100 build: adds DeepEP + transformer_engine + slime patches
export PATCH_VERSION="${PATCH_VERSION:-v0.5.9}"
case "${PATCH_VERSION}" in
  v0.5.9.a100) BUILD_A100=1 ;;
  v0.5.9)      BUILD_A100=0 ;;
  *) echo "Unknown PATCH_VERSION='${PATCH_VERSION}'"; exit 1 ;;
esac

export BASE_DIR=${BASE_DIR:-"/root"}
cd $BASE_DIR

# prevent installing cuda 13.0 for sglang
pip install cuda-python==13.1.0
PIP_DL="$SCRIPT_DIR/.downloads/pip-pkgs"
if [ -d "$PIP_DL" ]; then
  pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --find-links "$PIP_DL" --no-index
else
  pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
fi

# install sglang
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout ${SGLANG_COMMIT}
# Install the python packages
pip install -e "python[all]"


# DeepEP (only built for v0.5.9.a100 patch set)
if [ "$BUILD_A100" = "1" ]; then
  GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  cd $BASE_DIR
  git clone https://github.com/deepseek-ai/DeepEP.git
  cd DeepEP
  git checkout ${DEEPEP_COMMIT}
  # Apply deep_ep patch (loads libcuda.so RTLD_GLOBAL for driver API symbols)
  # Note: SLIME_DIR is not set yet at this point, locate the patch via the running script's repo.
  DEEPEP_PATCH="$SCRIPT_DIR/docker/patch/${PATCH_VERSION}/deep_ep.patch"
  if [ -f "$DEEPEP_PATCH" ]; then
    git apply "$DEEPEP_PATCH"
  fi
  if [ "${GPU_ARCH}" -ge 90 ] 2>/dev/null; then
    echo "GPU SM${GPU_ARCH} detected, installing DeepEP with NVSHMEM..."
    export NVSHMEM_DIR=$(python -c "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])" 2>/dev/null || true)
    # Create unversioned symlinks (e.g. libnvshmem_host.so -> libnvshmem_host.so.3)
    if [ -n "$NVSHMEM_DIR" ] && [ ! -f "$NVSHMEM_DIR/lib/libnvshmem_host.so" ]; then
      for f in "$NVSHMEM_DIR"/lib/*.so.*; do
        fname="$(basename "$f")"
        link="${fname%%.so.*}.so"
        [ ! -e "$NVSHMEM_DIR/lib/$link" ] && ln -sf "$fname" "$NVSHMEM_DIR/lib/$link"
      done
    fi
    TORCH_CUDA_ARCH_LIST="9.0" python setup.py install
  else
    echo "GPU SM${GPU_ARCH} (< SM90), installing DeepEP without SM90 features..."
    # torch pulls nvidia-nvshmem-cu12 which triggers false detection; force disable
    sed -i 's/    disable_nvshmem = False/    disable_nvshmem = True/' setup.py
    DISABLE_SM90_FEATURES=1 python setup.py install
    git checkout -- setup.py
  fi
  cd $BASE_DIR
fi

pip install cmake ninja

# flash attn
# the newest version megatron supports is v2.7.4.post1
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
# Apply transformer_engine patch to installed site-packages (a100 only)
if [ "$BUILD_A100" = "1" ]; then
  TE_SITE_DIR=$(python -c "import transformer_engine, os; print(os.path.dirname(transformer_engine.__path__[0]))")
  TE_PATCH="$SCRIPT_DIR/docker/patch/${PATCH_VERSION}/transformer_engine.patch"
  if patch -R -p1 -d "$TE_SITE_DIR" --dry-run < "$TE_PATCH" >/dev/null 2>&1; then
    echo "transformer_engine.patch already applied, skipping."
  else
    patch -p1 -d "$TE_SITE_DIR" < "$TE_PATCH"
  fi
fi
pip install flash-linear-attention==0.4.1
TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" \
  NVCC_APPEND_FLAGS="--threads 32" \
  TORCH_ALLOW_CUDA_MISMATCH=1 \
    pip -v install --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall

# megatron
cd $BASE_DIR
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
  cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
  pip install -e .

# install slime and apply patches
# if slime does not exist locally, clone it
if [ ! -d "$BASE_DIR/slime" ]; then
  cd $BASE_DIR
  cp -r $SCRIPT_DIR slime
  # git clone  https://github.com/THUDM/slime.git
  cd slime/
  export SLIME_DIR=$BASE_DIR/slime
  pip install -e .
else
  export SLIME_DIR=$BASE_DIR/
  pip install -e .
fi

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"

# apply patch
cd $BASE_DIR/sglang
git apply $SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch
cd $BASE_DIR/Megatron-LM
git apply $SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch
if [ "$BUILD_A100" = "1" ]; then
  cd $SLIME_DIR
  git apply $SLIME_DIR/docker/patch/${PATCH_VERSION}/slime.patch
fi

# final verification
python -c "
import torch; print(f'torch CUDA: {torch.version.cuda}')
import fused_weight_gradient_mlp_cuda; print('✅ gradient_accumulation_fusion OK')
import amp_C; print('✅ amp_C OK')
import transformer_engine; print(f'✅ TE {transformer_engine.__version__}')
import deep_ep; print('✅ DeepEP OK')
"