ARG SGLANG_VERSION=latest
FROM lmsysorg/sglang:${SGLANG_VERSION} AS sglang

# we need to write this again after from
ARG SGLANG_VERSION
ARG MEGATRON_COMMIT=core_v0.14.0

RUN apt update
RUN apt install -y nvtop rsync

# TODO: change to pip install sglang-router after it has a new release
RUN pip install sglang-router --force-reinstall
RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git --no-cache-dir --force-reinstall
RUN pip install ray[default]
RUN pip install httpx[http2] wandb pylatexenc blobfile accelerate "mcp[cli]"

# mbridge
RUN pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps

RUN TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;9.0a" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
# apex
RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git
# transformer engine, we install with --no-deps to avoid installing torch and torch-extensions
RUN pip install pybind11
# flash attn
# the newest version megatron supports is v2.7.4
RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1
RUN pip install flash-linear-attention
RUN pip -v install --no-build-isolation transformer_engine[pytorch]

RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git submodule update --init && cd hopper/ && python setup.py install && \
    export python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
    mkdir -p $python_path/flash_attn_3 && \
    cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py

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
