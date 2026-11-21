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
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0 10.3}"

# Keep these in sync with docker/Dockerfile:
#   - SGLANG_IMAGE_TAG (ARG)            -> SGLANG_VERSION below
#   - MEGATRON_COMMIT (ARG)             -> MEGATRON_COMMIT below
#   - PATCH_VERSION (ARG)               -> PATCH_VERSION below
#   - TMS_COMMIT (ARG)                   -> TMS_COMMIT below
#   - DEEPGEMM_COMMIT / DEEPEP_COMMIT     -> matching source builds below
export SGLANG_VERSION="v0.5.21"
export SGLANG_COMMIT="e00930c5489053f26d86b179cee0d087f846acbb"
export MEGATRON_COMMIT="ae76c60e6c35d913867d6dd7bad0f30b128b54e7"
export PATCH_VERSION="v0.5.21"
export TMS_COMMIT="bdab8c55f490a00a746ec34e301391617f5ca878"
export DEEPGEMM_COMMIT="8965e4a5cded4041c8dbbe8d35b0be31fd8da90b"
export DEEPEP_COMMIT="6845ffd9d59126ec0030c13e0e155935a61e5b5a"
export CUDA_COMPAT_VERSION="580.178.04-1ubuntu1"
export MOONCAKE_VERSION="0.3.11.post1"
export TRANSFORMER_ENGINE_VERSION="2.18.0"
export SLIME_NCCL_VERSION="2.30.7"

export BASE_DIR=${BASE_DIR:-"/root"}
cd "$BASE_DIR"

# Install the CUDA compiler and development libraries, without the full CUDA
# meta-package's Nsight, profilers and GUI tools. Runtime cuDNN and NCCL come
# from PyTorch's wheels below.
micromamba install -n slime \
  cuda-nvcc=13.0.88 \
  cuda-libraries-dev=13.0.3 \
  cuda-nvtx-dev=13.0.85 \
  -c nvidia/label/cuda-13.0.3 \
  -c nvidia \
  -c conda-forge \
  -y
# Extract only NVIDIA's user-space compatibility libraries into this environment.
case "$(uname -m)" in
  x86_64) compat_repo=x86_64; compat_arch=amd64 ;;
  aarch64) compat_repo=sbsa; compat_arch=arm64 ;;
  *) echo "Unsupported CUDA compatibility architecture" >&2; exit 1 ;;
esac
compat_tmp="$(mktemp -d)"
curl --fail --location --retry 3 \
  "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${compat_repo}/cuda-compat-13-0_${CUDA_COMPAT_VERSION}_${compat_arch}.deb" \
  -o "$compat_tmp/cuda-compat.deb"
dpkg-deb --extract "$compat_tmp/cuda-compat.deb" "$compat_tmp/package"
mkdir -p "$CONDA_PREFIX/cuda-compat"
cp -a "$compat_tmp/package/usr/local/cuda-13.0/compat/." "$CONDA_PREFIX/cuda-compat/"
rm -rf "$compat_tmp"
export SLIME_CUDA_COMPAT_PATH="$CONDA_PREFIX/cuda-compat"
source "$SLIME_DIR/docker/cuda-compat.sh"

# sglang's editable install builds a Rust extension (sglang-grpc via
# setuptools-rust), so the conda env needs a working rustc + cargo.
# DeepGEMM needs elfutils headers; DeepEP's NVSHMEM kernels need RDMA headers.
micromamba install -n slime -c conda-forge rust elfutils rdma-core -y

# Pin the CUDA 13 runtime before installing SGLang or Slime dependencies.
export PIP_CONSTRAINT="$CONDA_PREFIX/slime-constraints.txt"
cat > "$PIP_CONSTRAINT" <<'REQ'
torch==2.13.0
torchvision==0.28.0
torchaudio==2.11.0
cuda-python==13.4.1
cuda-bindings==13.4.3
sglang-kernel==0.4.7
sgl-deep-gemm==0.2.0
flash-attn-4==4.0.0b19
tilelang==0.1.12
numpy==2.3.5
scipy==1.18.1
pyarrow==25.0.1
kernels<0.15.0
setuptools<82
REQ
pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 torchaudio==2.11.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
pip install --no-deps sglang-kernel==0.4.7 sgl-deep-gemm==0.2.0
pip install cmake ninja wheel scikit-build-core "setuptools>=80.0.0"

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
# Build the fork before SGLang so its dependency does not install upstream TMS.
# Build isolation hides nvcc and produces a wheel without the preload hook.
pip install -v git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} \
  --no-cache-dir --no-build-isolation

# Install SGLang's LLM runtime. The [all] extra also pulls diffusion, video and
# tracing packages that are not used by slime's training/rollout environment.
if [ ! -d "$BASE_DIR/sglang" ]; then
  cd "$BASE_DIR"
  git clone https://github.com/sgl-project/sglang.git
fi
cd "$BASE_DIR/sglang"
git checkout ${SGLANG_COMMIT}
pip install -e python --extra-index-url https://download.pytorch.org/whl/cu130
pip install flash-linear-attention==0.5.2 tilelang==0.1.12
# 0.3.13 corrupts RDMA payloads after KV memory is remapped; match Docker.
pip install "mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
# Keep FA4 for Blackwell and add FA3 for Hopper MLA/context-parallel training.
pip install --no-deps flash-attn-3==3.0.0 --index-url https://download.pytorch.org/whl/cu130
# Use the FLA GDN backend on CUDA 13.

# Match Docker's official TE core wheel and compile only the PyTorch bindings.
# Use the pip NCCL and cuDNN libraries consistently instead of system copies.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
# SGLang JIT, DeepGEMM and PyTorch's FindCUDA expect the conventional toolkit
# layout. NVIDIA's conda packages use lib and put headers under targets.
if [ ! -e "$CONDA_PREFIX/lib64" ]; then
  ln -s lib "$CONDA_PREFIX/lib64"
fi
for cuda_header in "$CONDA_PREFIX/targets/${compat_repo}-linux/include/"*; do
  cuda_header_name="$(basename "$cuda_header")"
  if [ ! -e "$CONDA_PREFIX/include/$cuda_header_name" ]; then
    ln -s "../targets/${compat_repo}-linux/include/$cuda_header_name" "$CONDA_PREFIX/include/$cuda_header_name"
  fi
done
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/cccl:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/include:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/include${CPATH:+:$CPATH}"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/slime-nccl.sh" <<'SH'
export _SLIME_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export _SLIME_LD_LIBRARY_PATH_WAS_SET="${LD_LIBRARY_PATH+x}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export _SLIME_OLD_CUDA_COMPAT_PATH="${SLIME_CUDA_COMPAT_PATH-}"
export _SLIME_CUDA_COMPAT_PATH_WAS_SET="${SLIME_CUDA_COMPAT_PATH+x}"
export SLIME_CUDA_COMPAT_PATH="$CONDA_PREFIX/cuda-compat"
export _SLIME_OLD_CUDA_HOME="${CUDA_HOME-}"
export _SLIME_CUDA_HOME_WAS_SET="${CUDA_HOME+x}"
export CUDA_HOME="$CONDA_PREFIX"
source "$CONDA_PREFIX/etc/conda/slime-cuda-compat.sh"
SH
cp "$SLIME_DIR/docker/cuda-compat.sh" "$CONDA_PREFIX/etc/conda/slime-cuda-compat.sh"
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/slime-nccl.sh" <<'SH'
if [ -n "${_SLIME_LD_LIBRARY_PATH_WAS_SET:-}" ]; then
  export LD_LIBRARY_PATH="${_SLIME_OLD_LD_LIBRARY_PATH}"
else
  unset LD_LIBRARY_PATH
fi
if [ -n "${_SLIME_CUDA_COMPAT_PATH_WAS_SET:-}" ]; then
  export SLIME_CUDA_COMPAT_PATH="${_SLIME_OLD_CUDA_COMPAT_PATH}"
else
  unset SLIME_CUDA_COMPAT_PATH
fi
if [ -n "${_SLIME_CUDA_HOME_WAS_SET:-}" ]; then
  export CUDA_HOME="${_SLIME_OLD_CUDA_HOME}"
else
  unset CUDA_HOME
fi
unset _SLIME_OLD_LD_LIBRARY_PATH _SLIME_LD_LIBRARY_PATH_WAS_SET
unset _SLIME_OLD_CUDA_COMPAT_PATH _SLIME_CUDA_COMPAT_PATH_WAS_SET
unset _SLIME_OLD_CUDA_HOME _SLIME_CUDA_HOME_WAS_SET
SH
pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch
NVTE_WITH_NCCL_EP=0 MAX_JOBS=64 \
  pip install --no-build-isolation "transformer_engine[pytorch,core_cu13]==${TRANSFORMER_ENGINE_VERSION}"

NVCC_APPEND_FLAGS="--threads 1" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel ${MAX_JOBS}" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

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
# Megatron dev needs nvidia-resiliency-ext (its `dev` extra) for cached checkpoint metadata; keep the pin equal to Megatron's.
pip install --no-deps nvidia-resiliency-ext==0.7.0

# Install runtime dependencies before reasserting the compatibility pins.

cd "$SLIME_DIR"
# Install slime's pure-python runtime deps first (wandb, ray, accelerate,
# transformers, etc.) from its requirements.txt, then install slime itself
# with --no-deps so pip doesn't re-resolve and stomp our pinned native libs
# (torch+cu130, sglang-kernel, ...). The Dockerfile does the same thing
# before the editable package installation.
pip install -r requirements.txt "sglang==${SGLANG_VERSION#v}"
pip install -e . --no-deps

# int4_qat kernel (matches Dockerfile)
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
pip install . --no-build-isolation --no-deps

pip install "numpy==2.3.5" "scipy==1.18.1" "pyarrow==25.0.1"
# kernels 0.15.x trips a ValueError("Either a revision or a version must be
# specified") on `transformers.integrations.hub_kernels` import; pin to <0.15
# so `import sglang` works at runtime.
pip install "kernels<0.15.0"

# Constraints retain the CUDA 13 stack throughout dependency resolution. Only
# the router fork and NCCL override need installing after the runtime deps.
pip install --no-deps https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
pip install --no-deps "nvidia-nccl-cu13==${SLIME_NCCL_VERSION}"
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

cd "$BASE_DIR"
if [ ! -d DeepGEMM ]; then
  git clone https://github.com/zhuzilin/DeepGEMM.git --recursive
fi
cd DeepGEMM
git checkout "$DEEPGEMM_COMMIT"
git submodule update --init --recursive
bash build_sgl_deep_gemm.sh
pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl

# The upstream sgl-deep-ep wheel shares this Python namespace with the Slime fork.
# Remove it after all resolving installs, before replacing the package.
pip uninstall -y sgl-deep-ep deep-ep
cd "$BASE_DIR"
if [ ! -d DeepEP ]; then
  git clone https://github.com/zhuzilin/DeepEP.git
fi
cd DeepEP
git checkout "$DEEPEP_COMMIT"
if [ -d "$CUDA_HOME/include/cccl" ]; then
  export CPATH="$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
  export NVCC_PREPEND_FLAGS="-I$CUDA_HOME/include/cccl ${NVCC_PREPEND_FLAGS:-}"
fi
python setup.py bdist_wheel
pip install --force-reinstall --no-deps dist/deep_ep-*.whl

python - <<'PY'
from importlib.metadata import version
import ctypes

import sglang
import torch
import torchaudio
import torchvision
import pyarrow
import transformer_engine.pytorch
from megatron.core import parallel_state

assert torch.__version__.split("+")[0] == "2.13.0"
assert torchaudio.__version__.split("+")[0] == "2.11.0"
assert torchvision.__version__.split("+")[0] == "0.28.0"
assert torch.version.cuda == "13.0"
nccl_version = ctypes.c_int()
assert ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(nccl_version)) == 0
assert nccl_version.value == 23007
assert version("nvidia-nccl-cu13") == "2.30.7"
assert version("sglang") == "0.5.21"
assert version("sglang-kernel").split("+")[0] == "0.4.7"
assert version("sgl-deep-gemm").split("+")[0] == "0.2.0"
assert version("cuda-python") == "13.4.1"
assert version("numpy") == "2.3.5"
assert version("scipy") == "1.18.1"
assert pyarrow.__version__ == "25.0.1"
assert version("tilelang") == "0.1.12"
assert version("megatron-core").split("+")[0] == "0.20.0"
assert version("transformer-engine") == "2.18.0"
assert version("flash-attn-4").startswith("4.")
assert version("flash-attn-3") == "3.0.0"
assert version("mooncake-transfer-engine-cuda13") == "0.3.11.post1"
import deep_gemm
assert callable(deep_gemm.set_batch_invariant)
import deep_ep
assert hasattr(deep_ep, "Buffer")
assert version("deep-ep") == "1.1.0+6845ffd"
assert version("flash-linear-attention") == "0.5.2"
assert version("fla-core") == "0.5.2"
assert hasattr(torch.ops.torchvision, "nms")
PY
