ARG SGLANG_IMAGE_TAG=v0.5.21
ARG SGLANG_IMAGE=lmsysorg/sglang
FROM ${SGLANG_IMAGE}:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Arguments =============================================

ARG PATCH_VERSION=v0.5.21
ARG MEGATRON_COMMIT=ae76c60e6c35d913867d6dd7bad0f30b128b54e7
ARG DEEPGEMM_COMMIT=8965e4a5cded4041c8dbbe8d35b0be31fd8da90b
ARG DEEPEP_COMMIT=6845ffd9d59126ec0030c13e0e155935a61e5b5a
ARG DEEPEP_CUDA_ARCH_LIST="9.0 10.0 10.3"
ARG TRANSFORMER_ENGINE_VERSION=2.18.0
ARG SLIME_NCCL_VERSION=2.30.7
ARG MOONCAKE_VERSION=0.3.11.post1
ARG CUDA_COMPAT_VERSION=580.178.04-1ubuntu1
ARG TMS_COMMIT=bdab8c55f490a00a746ec34e301391617f5ca878

ARG TMS_CUDA_MAJOR=
ARG MAX_JOBS=16
ENV MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0 10.3"
# Resolve NCCL consistently for PyTorch, SGLang and the official TE core wheel.
ENV LD_LIBRARY_PATH=/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt-get update && apt-get install -y --no-install-recommends \
      nvtop rsync dnsutils prometheus libdw-dev libelf-dev "cuda-compat-13-0=${CUDA_COMPAT_VERSION}" && \
    rm -rf /var/lib/apt/lists/*
COPY docker/cuda-compat.sh docker/entrypoint.sh /opt/slime/
RUN chmod +x /opt/slime/entrypoint.sh
ENTRYPOINT ["/opt/slime/entrypoint.sh"]

# ====================================== Python dependencies ============================================

# Preserve SGLang's CUDA 13/PyTorch stack during later resolving installs.
RUN python -c 'import importlib.metadata as m; packages = ("torch", "torchvision", "torchaudio", "cuda-python", "cuda-bindings", "sglang-kernel", "sgl-deep-gemm", "flashinfer-python", "flashinfer-cubin", "transformers"); open("/opt/slime/constraints.txt", "w").write("".join(f"{p}=={m.version(p)}\n" for p in packages) + "pyarrow==25.0.1\n")'
ENV PIP_CONSTRAINT=/opt/slime/constraints.txt
RUN pip install flash-linear-attention==0.5.2 tilelang==0.1.12 scikit-build-core
# Keep FA4 for Blackwell and add FA3 for Hopper MLA/context-parallel training.
RUN pip install --no-deps flash-attn-3==3.0.0 --index-url https://download.pytorch.org/whl/cu130
# Use the FLA GDN backend on CUDA 13.

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

# Megatron dev reads cached checkpoint metadata (slime sets ckpt_assume_constant_structure)
# and runs async checkpointing through nvidia-resiliency-ext, which sits in its `dev` extra
# and is therefore not pulled in by `pip install -e .`. Keep the pin equal to Megatron's.
RUN pip install --no-deps nvidia-resiliency-ext==0.7.0

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

RUN pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt "sglang==$(python -c 'import importlib.metadata as m; print(m.version("sglang"))')"

# Match the base scientific Python stack; CuPy's CUDA 13 wheel requires NumPy 2.
RUN pip install "numpy==2.3.5" "scipy==1.18.1" "pyarrow==25.0.1" "kernels<0.15.0"

# 0.3.13 can report success with a corrupt RDMA payload after KV memory is
# remapped by torch_memory_saver. Keep the version validated by the PD tests.
RUN pip install --no-deps --force-reinstall "mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"

# requirements.txt installs sglang-router and its dependencies above. Replace
# only the router with the fork wheel; re-resolving its unbounded dependencies
# here can trigger extensive aiohttp backtracking and alter the existing stack.
RUN pip install --no-deps --force-reinstall https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl
RUN python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# DeepEP low-latency alignment fork.  CUDA 13 relocated the CCCL/libcu++ headers
# (<cuda/std/*>) under include/cccl, which nvshmem_tensor.h includes, so add that
# directory to the compile search path (nvcc + host compiler) when it exists.
# Install after resolving dependencies: sgl-deep-ep shares the deep_ep namespace.
# Keep this before patches and Slime sources so their changes reuse the build.
RUN pip uninstall -y sgl-deep-ep deep-ep && \
    git clone https://github.com/zhuzilin/DeepEP.git /root/DeepEP && \
    cd /root/DeepEP && git checkout ${DEEPEP_COMMIT} && \
    CUDA_ARCHS="${DEEPEP_CUDA_ARCH_LIST}" && \
    if [ -d /usr/local/cuda/include/cccl ]; then \
      export CPATH="/usr/local/cuda/include/cccl${CPATH:+:$CPATH}"; \
      export NVCC_PREPEND_FLAGS="-I/usr/local/cuda/include/cccl ${NVCC_PREPEND_FLAGS:-}"; \
    fi && \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" python setup.py bdist_wheel && \
    pip install --force-reinstall --no-deps dist/deep_ep-*.whl && \
    cd /root/ && rm -rf DeepEP

# PyTorch's dependency metadata pins older NCCL. Restore the tested runtime
# after resolving dependencies, before the frequently changed patches/sources.
# The remaining Slime and int4 kernel installs do not resolve runtime dependencies.
RUN CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" && \
    pip install --no-deps --force-reinstall "nvidia-nccl-cu${CUDA_MAJOR}==${SLIME_NCCL_VERSION}"

RUN rm -rf /root/.cache/pip

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

ARG SLIME_COMMIT=local
COPY . /root/slime/
RUN if [ "$SLIME_COMMIT" != "local" ]; then \
      rm -rf /root/slime && \
      git clone https://github.com/THUDM/slime.git /root/slime && \
      git -C /root/slime checkout "$SLIME_COMMIT"; \
    fi && pip install -e /root/slime --no-deps

RUN cd /root/slime/slime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation --no-deps
