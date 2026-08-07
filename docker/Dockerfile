ARG SGLANG_IMAGE_TAG=v0.5.15.post1-cu129
FROM slimerl/sglang:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG DEEPGEMM_COMMIT=b38a77cd193cf38f670caae192310521d24343be
ARG DEEPEP_COMMIT=6845ffd9d59126ec0030c13e0e155935a61e5b5a
ARG DEEPEP_CUDA_ARCH_LIST=
ARG FLASH_QLA_COMMIT=821fd9d37ede18fdc2a4e707fefe3770bfc32e58
ARG TRANSFORMER_ENGINE_COMMIT=c9877beb87ad7e711e1869dd0b5062167ede447a
ARG TRANSFORMER_ENGINE_CUDA_ARCHS=100a;103a
ARG TMS_COMMIT=8d30c59ca12a68d9deccbc9c6599076a1218cbc5

ARG TMS_CUDA_MAJOR=

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt update
RUN apt install -y nvtop rsync dnsutils prometheus

# ====================================== Python dependencies ============================================

# CUDA 12 training uses FA2/FA3.  On CUDA 13, keep the FA4 package supplied by
# SGLang instead of replacing its Blackwell inference kernels with FA2.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "12" ]; then \
      pip uninstall -y flash-attn-4 flash_attn_4 || true; \
      MAX_JOBS=64 pip -v install flash-attn==2.8.3 --no-build-isolation; \
      git clone https://github.com/Dao-AILab/flash-attention.git && \
      cd flash-attention/ && git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c && git submodule update --init && \
      cd hopper/ && \
      FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 pip -v install . --no-build-isolation && \
      cd /root/ && rm -rf flash-attention/; \
    fi

RUN pip install flash-linear-attention==0.4.2
# FlashQLA currently requires TileLang 0.1.9, while SGLang 0.5.15.post1 on
# CUDA 13 requires 0.1.11.  Keep the existing CUDA 12 installation and use
# the default FLA backend on CUDA 13 instead of downgrading SGLang's runtime.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "12" ]; then \
      pip install \
        git+https://github.com/QwenLM/FlashQLA.git@${FLASH_QLA_COMMIT} \
        --no-build-isolation; \
      pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/; \
    fi

# The cu13 TE wheel is built against a newer cuBLAS than SGLang's CUDA 13.0
# toolkit.  Build TE against the base image's libraries on CUDA 13; retain the
# validated wheel path for CUDA 12.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "13" ]; then \
      NVTE_CUDA_ARCHS="${TRANSFORMER_ENGINE_CUDA_ARCHS}" MAX_JOBS=64 \
        pip -v install --no-build-isolation \
        git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_COMMIT}; \
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.16.1"; \
    fi

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT} && \
    pip install -e .

# zhuzilin fork builds, grouped together right after Megatron-LM:
# torch_memory_saver, plus the GLM-5 train/rollout alignment kernels
# (DeepGEMM batch-invariant selection and the DeepEP low-latency fork).
RUN TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}" && \
    export TMS_CUDA_MAJOR && \
    pip install git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} --no-cache-dir --force-reinstall

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
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" MAX_JOBS=64 python setup.py bdist_wheel && \
    pip install --force-reinstall --no-deps dist/deep_ep-*.whl && \
    cd /root/ && rm -rf DeepEP

RUN pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

# Megatron currently requires NumPy 1.x, while SciPy 1.18+ requires NumPy 2.x.
# Pin both packages together so downgrading NumPy does not leave an incompatible SciPy.
RUN pip install "numpy==1.26.4" "scipy==1.17.1"

RUN pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
RUN python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Patches ============================================

COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    git update-index --refresh && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron.patch

# TODO temporarily skip patching for GB200/GB300 (and require users to bring their own sglang version). should add back later.
ARG ENABLE_SGLANG_PATCH=1
COPY docker/patch/${PATCH_VERSION}/sglang*.patch /sgl-workspace/sglang/
RUN if [ "$ENABLE_SGLANG_PATCH" = "1" ]; then \
  cd /sgl-workspace/sglang && \
  for patch in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do \
    if [ ! -f "$patch" ]; then \
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
