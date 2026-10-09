ARG SGLANG_IMAGE_TAG=v0.5.15.post1-cu129
FROM slimerl/sglang:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Arguments =============================================

ARG PATCH_VERSION=v0.5.15.post1
ARG MEGATRON_COMMIT=4b4acac9a1d28ea6829c8d4f566d75698a21249d
ARG DEEPGEMM_COMMIT=b38a77cd193cf38f670caae192310521d24343be
ARG DEEPEP_COMMIT=6845ffd9d59126ec0030c13e0e155935a61e5b5a
ARG DEEPEP_CUDA_ARCH_LIST=
ARG FLASH_QLA_COMMIT=821fd9d37ede18fdc2a4e707fefe3770bfc32e58
ARG TRANSFORMER_ENGINE_VERSION=2.18.0
ARG NCCL_VERSION=2.30.7
ARG TMS_COMMIT=4d525cf378fdbfe7eb044909b8e66e112a508624

ARG TMS_CUDA_MAJOR=
ARG MAX_JOBS=16
ENV MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}
# Resolve NCCL consistently for PyTorch, SGLang and the official TE core wheel.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt update
RUN apt install -y nvtop rsync dnsutils prometheus

# ====================================== Python dependencies ============================================

# TE 2.18 supports FA2 through 2.8.3. These wheels match Python 3.12,
# PyTorch 2.11 and CUDA 12; FA3 comes from the official PyTorch wheel index.
# On CUDA 13, keep the base image's FA4 kernels for Blackwell.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "12" ]; then \
      pip uninstall -y flash-attn-4 flash_attn_4 || true; \
      pip install --no-deps --force-reinstall \
        "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
        "https://download.pytorch.org/whl/cu129/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"; \
    fi

RUN pip install flash-linear-attention==0.5.2
# FlashQLA currently requires TileLang 0.1.9, while SGLang 0.5.15.post1 on
# CUDA 13 requires 0.1.11.  Keep the existing CUDA 12 installation and use
# the default FLA backend on CUDA 13 instead of downgrading SGLang's runtime.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "12" ]; then \
      pip install \
        git+https://github.com/QwenLM/FlashQLA.git@${FLASH_QLA_COMMIT} \
        --no-build-isolation; \
      pip install tilelang==0.1.9 -f https://tile-ai.github.io/whl/nightly/cu128/; \
    fi

# Use the official TE core wheel; only its PyTorch bindings need compilation.
# Slime uses DeepEP, so leave the optional NCCL EP bindings disabled.
RUN CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" && \
    pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch && \
    NVTE_WITH_NCCL_EP=0 MAX_JOBS=64 \
      pip -v install --no-build-isolation \
      "transformer_engine[pytorch,core_cu${CUDA_MAJOR}]==${TRANSFORMER_ENGINE_VERSION}"

RUN NVCC_APPEND_FLAGS="--threads 1" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel ${MAX_JOBS}" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT} && \
    pip install -e .

# zhuzilin fork builds, grouped together right after Megatron-LM:
# torch_memory_saver, plus the GLM-5 train/rollout alignment kernels
# (DeepGEMM batch-invariant selection and the DeepEP low-latency fork).
RUN TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}" && \
    export TMS_CUDA_MAJOR && \
    pip install git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} --no-cache-dir --force-reinstall --no-build-isolation

# DeepGEMM's batch-invariant kernel selection; build the SGLang-compatible wheel.
RUN git clone https://github.com/zhuzilin/DeepGEMM.git --recursive && \
    cd DeepGEMM && git checkout ${DEEPGEMM_COMMIT} && \
    git submodule update --init --recursive && \
    bash build_sgl_deep_gemm.sh && \
    pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl && \
    cd /root/ && rm -rf DeepGEMM

# DeepEP low-latency alignment fork.  CUDA 13 relocated the CCCL/libcu++ headers
# (<cuda/std/*>) under include/cccl, which nvshmem_tensor.h includes, so add that
# directory to the compile search path (nvcc + host compiler) when it exists.
RUN git clone https://github.com/zhuzilin/DeepEP.git /root/DeepEP && \
    cd /root/DeepEP && git checkout ${DEEPEP_COMMIT} && \
    CUDA_ARCHS="${DEEPEP_CUDA_ARCH_LIST:-9.0}" && \
    if [ -d /usr/local/cuda/include/cccl ]; then \
      export CPATH="/usr/local/cuda/include/cccl${CPATH:+:$CPATH}"; \
      export NVCC_PREPEND_FLAGS="-I/usr/local/cuda/include/cccl ${NVCC_PREPEND_FLAGS:-}"; \
    fi && \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" python setup.py bdist_wheel && \
    pip install --force-reinstall --no-deps dist/deep_ep-*.whl && \
    cd /root/ && rm -rf DeepEP

RUN pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

# Megatron currently requires NumPy 1.x, while SciPy 1.18+ requires NumPy 2.x.
# Pin both packages together so downgrading NumPy does not leave an incompatible SciPy.
RUN pip install "numpy==1.26.4" "scipy==1.17.1" "kernels<0.15.0"

# requirements.txt installs sglang-router and its dependencies above. Replace
# only the router with the fork wheel; re-resolving its unbounded dependencies
# here can trigger extensive aiohttp backtracking and alter the existing stack.
RUN pip install --no-deps --force-reinstall https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl
RUN python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Patches ============================================

COPY docker/patch/${PATCH_VERSION}/megatron*.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    for patch_name in megatron.patch megatron-sglang-aligned.patch; do \
      if [ ! -f "$patch_name" ]; then \
        if [ "$patch_name" = "megatron.patch" ]; then \
          echo "Required patch is missing: $patch_name" >&2; \
          exit 1; \
        fi; \
        continue; \
      fi; \
      git update-index --refresh && \
        git apply "$patch_name" --3way || exit 1; \
      if git grep -n '^<<<<<<< ' -- .; then \
        echo "$patch_name failed to apply cleanly. Please resolve conflicts." >&2; \
        exit 1; \
      fi; \
    done && \
    rm -f megatron*.patch

# Custom base images can opt out of the SGLang patch stack.
ARG ENABLE_SGLANG_PATCH=1
COPY docker/patch/${PATCH_VERSION}/sglang*.patch /sgl-workspace/sglang/
RUN if [ "$ENABLE_SGLANG_PATCH" = "1" ]; then \
  cd /sgl-workspace/sglang && \
  for patch in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch sglang-deterministic.patch; do \
    if [ ! -f "$patch" ]; then \
      if [ "$patch" = "sglang.patch" ]; then \
        echo "Required patch is missing: $patch" >&2; \
        exit 1; \
      fi; \
      continue; \
    fi; \
    if ! git apply --check "$patch"; then \
      echo "Patch $patch does not apply to the SGLang source in the base image."; \
      exit 1; \
    fi; \
    git apply "$patch" || exit 1; \
  done && \
  rm sglang*.patch; \
fi

# ====================================== Install main package ============================================

ARG SLIME_COMMIT=main
RUN git clone https://github.com/THUDM/slime.git /root/slime && \
    cd /root/slime && \
    git checkout ${SLIME_COMMIT} && \
    pip install -e . --no-deps

RUN cd /root/slime/slime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation

# PyTorch's dependency metadata pins older NCCL. Restore the tested runtime
# after all resolving installs, without changing PyTorch or other packages.
RUN CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" && \
    pip install --no-deps --force-reinstall "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION}"
