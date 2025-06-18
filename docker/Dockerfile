FROM nvcr.io/nvidia/pytorch:25.04-py3 AS base

WORKDIR /root/
RUN apt update && apt install tmux -y

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip install torchao fastapi compressed-tensors datasets \
        hf_transfer huggingface_hub interegular "llguidance>=0.7.11,<0.8.0" \
        modelscope ninja orjson packaging partial_json_parser \
        pillow "prometheus-client>=0.20.0" \
        psutil pydantic pynvml python-multipart "pyzmq>=25.1.2" "soundfile==0.13.1" \
        transformers uvicorn uvloop xgrammar blobfile scikit_build_core uv

# sglang
RUN git clone https://github.com/sgl-project/sglang.git --branch v0.4.7 && \
    cd sglang/ && pip install -e "python[all]" --no-deps

# sgl-kernel
RUN cd sglang/sgl-kernel/ && MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=24 TORCH_CUDA_ARCH_LIST="9.0;9.0a" make build

# patch
COPY patch/sglang.patch /root/sglang/
RUN cd sglang/ && git apply sglang.patch && rm sglang.patch

# sgl-router
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN pip install setuptools-rust wheel build
RUN source $HOME/.cargo/env && cd sglang/sgl-router/ && \
    cargo build -j 64 && python3 -m build && pip install dist/*.whl --force-reinstall

RUN pip install torch_memory_saver

# deepgemm
RUN pip install cuda-bindings && \
    git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git && \
    cd DeepGEMM && \
    python setup.py install

RUN TORCH_CUDA_ARCH_LIST="9.0" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

# flashinfer
RUN rm -rf /usr/include/cutlass && rm /etc/pip/constraint.txt && touch /etc/pip/constraint.txt && pip install packaging -U
RUN git clone https://github.com/flashinfer-ai/flashinfer.git --recursive flashinfer-main && \
    cd flashinfer-main && \
    TORCH_CUDA_ARCH_LIST="9.0;9.0a" FLASHINFER_ENABLE_AOT=1 pip install --no-build-isolation -v -e .

RUN pip install ray[default] pylatexenc blobfile accelerate "mcp[cli]" openai

# vllm
RUN pip install msgspec watchfiles blake3 depyf lark lm-format-enforcer "mistral_common[opencv]>=1.5.4" "opencv-python-headless>=4.11.0" gguf py-cpuinfo
RUN pip install vllm --no-deps

# buffer
RUN pip install multiprocess timeout_decorator nvitop --no-deps
RUN pip install httpx[http2] wandb

# mbridge
RUN pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps

# to fix cannot install blinker error
RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && \
    pip install -e .

# sandwitch norm for GLM models
COPY patch/megatron-sandwich_norm.patch /root/Megatron-LM/
RUN cd Megatron-LM && git apply megatron-sandwich_norm.patch && rm megatron-sandwich_norm.patch