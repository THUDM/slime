ARG SGLANG_IMAGE_TAG=v0.5.5.post1
FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=core_v0.14.0

ARG ENABLE_CUDA_13=0

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt update
RUN apt install -y nvtop rsync

# ====================================== Python dependencies ============================================

# The compilation is slow, thus should be put at top
# TransformerEngines does not support too high FA2
RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

# The compilation is slow, thus should be put at top
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89 && git submodule update --init && cd hopper/ && python setup.py install && \
    export python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
    mkdir -p $python_path/flash_attn_3 && \
    cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py && \
    rm -rf flash-attention/

RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

RUN pip install flash-linear-attention==0.4.0

# TE does not have wheel on cuda 13 yet, thus need to install from source
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install nvidia-mathdx==25.6.0 && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.8; \
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.8.0"; \
    fi

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT} && \
    pip install -e .

RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@383bec3919fc266b25bdaaa0447ebd1ee9275f1e --no-cache-dir --force-reinstall

# This patch from masahi will be included in later Triton releases
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    (cd /root && git clone -b feat/v350_plus_8045 https://github.com/fzyzcjy/triton.git && cd triton && pip install -r python/requirements.txt && pip install --verbose -e .); \
  fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Temporarily install another sgl-kernel version for GB300 without rebuilding the whole image
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    SGL_KERNEL_VERSION=0.3.17 && \
    python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps; \
  fi

# This patch is merged into main, but we are using stable version, thus still need it
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    curl -L https://github.com/NVIDIA/TransformerEngine/pull/2286.patch -o /root/te2286.patch && (cd /usr/local/lib/python3.12/dist-packages/transformer_engine && (patch -p2 < /root/te2286.patch)); \
  fi

RUN rm /root/.tmux.conf

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

# TODO temporarily skip patching for GB300 (and require users to bring their own sglang version). should add back later.
COPY docker/patch/${PATCH_VERSION}/sglang.patch /sgl-workspace/sglang/
RUN if [ "$ENABLE_CUDA_13" = "0" ]; then \
  cd /sgl-workspace/sglang && \
  git update-index --refresh && \
  git apply sglang.patch && \
  if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
  fi && \
  rm sglang.patch; \
fi

# ====================================== Install main package ============================================

# TODO may improve
ARG SLIME_COMMIT=main
RUN git clone https://github.com/slimerl/slime /root/slime && \
    cd /root/slime && \
    git checkout ${SLIME_COMMIT} && \
    pip install -e .
