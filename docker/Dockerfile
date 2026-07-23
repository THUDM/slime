ARG SGLANG_IMAGE_TAG=v0.5.15.post1-cu129
FROM slimerl/sglang:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087

ARG ENABLE_CUDA_13=0
ARG TMS_CUDA_MAJOR=

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt update
RUN apt install -y nvtop rsync dnsutils prometheus

# ====================================== Python dependencies ============================================

# The SGLang base image may include FA4.  The validated TE 2.16 CP stack uses
# FA2 + FA3, so remove FA4 explicitly to avoid backend selection conflicts.
RUN pip uninstall -y flash-attn-4 flash_attn_4 || true

# The compilation is slow, thus should be put at top
RUN MAX_JOBS=64 pip -v install flash-attn==2.8.3 --no-build-isolation

# The compilation is slow, thus should be put at top.  This FA3 commit uses the
# window_size_left/window_size_right API required by TransformerEngine 2.16 CP.
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c && git submodule update --init && \
    cd hopper/ && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 pip -v install . --no-build-isolation && \
    cd /root/ && rm -rf flash-attention/

RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

RUN pip install flash-linear-attention==0.4.2
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
RUN pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# TE does not have wheel on cuda 13 yet, thus need to install from source
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install nvidia-mathdx==26.6.0 && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.16; \
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

RUN TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}" && \
    export TMS_CUDA_MAJOR && \
    pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 --no-cache-dir --force-reinstall
RUN pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# This patch from masahi will be included in later Triton releases
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    (cd /root && git clone -b feat/v350_plus_8045 https://github.com/fzyzcjy/triton.git && cd triton && pip install -r python/requirements.txt && pip install --verbose -e .); \
  fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

# Temporarily install another sgl-kernel version for GB300 without rebuilding the whole image
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    SGL_KERNEL_VERSION=0.3.17.post2 && \
    python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps; \
  fi

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
