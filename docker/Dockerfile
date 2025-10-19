ARG SGLANG_VERSION=latest
# TODO will unify after it is merged into sglang main
# FROM lmsysorg/sglang:${SGLANG_VERSION} AS sglang
FROM ishandhanani/sglang:gb300-tom-deepep AS sglang

# we need to write this again after from
ARG SGLANG_VERSION
ARG MEGATRON_COMMIT=ca9797e95e88d93623b0f69831529a1b521297ac

ARG ENABLE_BLACKWELL_BUILD=0
ARG ENABLE_CUDA_13=0
ARG ENABLE_TIGHT_BUILD_RES=0

RUN apt update
RUN apt install -y nvtop

# TODO: change to pip install sglang-router after it has a new release
RUN pip install sglang-router --force-reinstall
RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git --no-cache-dir --force-reinstall
RUN pip install ray[default]
RUN pip install httpx[http2] wandb pylatexenc blobfile accelerate "mcp[cli]"

# mbridge
RUN pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps

RUN if [ "${ENABLE_BLACKWELL_BUILD}" = "1" ]; then \
          export TLIST="8.0;8.9;9.0;9.0a;10.0"; \
        else \
          export TLIST="8.0;8.9;9.0;9.0a"; \
        fi && \
    TORCH_CUDA_ARCH_LIST="$TLIST" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

# apex
RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git
# transformer engine, we install with --no-deps to avoid installing torch and torch-extensions
RUN pip install pybind11

# flash attn
# the newest version megatron supports is v2.7.4
RUN if [ "$ENABLE_TIGHT_BUILD_RES" = "1" ]; then \
      export MAX_JOBS=32; \
    else \
      export MAX_JOBS=64; \
    fi && \
    pip -v install flash-attn==2.7.4.post1

RUN pip install flash-linear-attention

# TE does not have wheel on cuda 13 yet, thus need to install from source
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
          pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable; \
        else \
          pip -v install --no-build-isolation "transformer_engine[pytorch]"; \
        fi

WORKDIR /root/
RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && \
    pip install -e .

# remove the cudnn installed by megatron
RUN pip uninstall nvidia-cudnn-cu12 -y

# sandwitch norm for GLM models
COPY patch/${SGLANG_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    git checkout ${MEGATRON_COMMIT} && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron.patch

# sglang patch
COPY patch/${SGLANG_VERSION}/sglang.patch /sgl-workspace/sglang/
RUN cd /sgl-workspace/sglang && \
  git apply sglang.patch && \
  if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
  fi && \
  rm sglang.patch

RUN rm /root/.tmux.conf
