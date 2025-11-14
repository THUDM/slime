ARG SGLANG_VERSION=latest
FROM lmsysorg/sglang:${SGLANG_VERSION} AS sglang

# we need to write this again after from
ARG SGLANG_VERSION
ARG MEGATRON_COMMIT=core_v0.14.0

RUN apt update
RUN apt install -y nvtop rsync

# TODO: change to pip install sglang-router after it has a new release
RUN pip install sglang-router==0.2.1 --force-reinstall
RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@383bec3919fc266b25bdaaa0447ebd1ee9275f1e --no-cache-dir --force-reinstall
RUN pip install ray[default]
RUN pip install httpx[http2] wandb pylatexenc blobfile accelerate "mcp[cli]"

# mbridge
RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

# apex
RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
# transformer engine, we install with --no-deps to avoid installing torch and torch-extensions
RUN pip install pybind11
# flash attn
# the newest version megatron supports is v2.7.4
RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation
RUN pip install flash-linear-attention==0.4.0
RUN pip -v install --no-build-isolation 'transformer_engine[pytorch]==2.9.0'

WORKDIR /root/
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89 && git submodule update --init && cd hopper/ && python setup.py install && \
    export python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
    mkdir -p $python_path/flash_attn_3 && \
    cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py
RUN rm -rf flash-attention/

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT} && \
    pip install -e .

# sandwitch norm for GLM models
COPY patch/${SGLANG_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
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
