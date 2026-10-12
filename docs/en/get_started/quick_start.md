# Environment setup

Start in the {ref}`interactive Quick Start <lab>`. This page prepares the environment used by every generated recipe; the [tutorial](experiment-guide.md) explains the full RL workflow.

## Basic Environment Setup

Use the provided Docker images to get a compatible SGLang, Megatron, and patch stack.

### Hardware and Image Selection

| Hardware | Image | CUDA |
| --- | --- | --- |
| H100/H200 | `slimerl/slime:latest` or `slimerl/slime:latest-cu129` | 12.9 |
| B200/B300 (Blackwell, x86) | `slimerl/slime:latest-cu130` | 13.0 |

The default `latest` tag follows the CUDA 12 image. For Blackwell, replace the image in the commands below with `slimerl/slime:latest-cu130`. The two variants use different kernel and Transformer Engine builds. See the [Docker guide](https://github.com/THUDM/slime/blob/main/docker/README.md) for source pins and build options.

GPU CI primarily covers H100/H200. Check each recipe's hardware and parallelism requirements before running it on other platforms.

For installation without Docker, use the {ref}`conda path below <conda>`. For AMD, see the [AMD tutorial](../platform_support/amd_tutorial.md).

### Pull and Start Docker Container

On a Linux x86_64 GPU host, prepare Git, Docker, a compatible NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Run these commands on the **host** to pull the image, clone slime and start a container:

```shell
SLIME_IMAGE=slimerl/slime:latest-cu129
docker pull "$SLIME_IMAGE"
git clone https://github.com/THUDM/slime.git
cd slime
mkdir -p ../slime-data
docker run --rm -it --gpus all --ipc=host \
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/root/slime" -v "$PWD/../slime-data:/data" \
  -w /root/slime "$SLIME_IMAGE" /bin/bash
```

### Install slime

The image already contains slime and its dependencies. The bind mount replaces the image's checkout with your host checkout. Run the following **inside the container** to install that checkout without replacing the image's pinned dependencies, then check imports and GPU visibility:

```bash
pip install -e . --no-deps
python -c "import slime, sglang, megatron.core; import torch; print('GPUs:', torch.cuda.device_count())"
```

The GPU count should match the devices exposed to the container. Save the builder's `experiment.sh` in the host `slime` directory; it is immediately visible at `/root/slime/experiment.sh` inside the container. Models, data and checkpoints written under `/data` persist in the host's sibling `slime-data` directory after the container exits. To reopen the environment, repeat the `docker run` command from the same checkout and the editable install inside the new container.

(conda)=
## Conda / micromamba

[build_conda.sh](https://github.com/THUDM/slime/blob/main/build_conda.sh) installs micromamba and creates the `slime` environment with Python 3.12 and CUDA 12.9. It builds Megatron, SGLang and native dependencies, installs slime, and applies the repository's matching patches. This takes longer than starting a prepared image.

Use a Linux x86_64 NVIDIA machine with Bash, Git, curl, patch, a C/C++ toolchain and a compatible GPU driver. The current script writes to `/root` (including `/root/.cargo`) and expects permission to do so; the commands below use its default `/root` source directory. For B200/B300, use the CUDA 13 Docker image instead.

From a Bash shell:

```bash
git clone https://github.com/THUDM/slime.git
cd slime
BASE_DIR=/root SHELL=/bin/bash bash build_conda.sh
```

After the build succeeds, activate the environment in your current shell. Repeat these activation commands in each new Bash session:

```bash
source ~/.bashrc
micromamba activate slime
export CUDA_HOME="$CONDA_PREFIX"
python -c "import slime, sglang, megatron.core; import torch; print('GPUs:', torch.cuda.device_count())"
```

Keep using your slime checkout to run the generated script. In the builder's **Files** step, set model, data and output paths to writable directories on this machine. The Megatron path is `/root/Megatron-LM` with the setup above.

## Mount the experiment paths

The Docker commands above mount a persistent data directory at `/data`, matching the builder's defaults. To use an existing dataset or shared storage, replace that mount with, for example, `-v /shared/data:/data`, and use the container-visible paths in the lab. Keep the repository and generated configuration directory visible at the same path on all Ray hosts. Install the same dependencies on every node and configure networking so the Ray nodes can reach each other; the single-host Docker command above does not set up a multi-node network.

Return to the [tutorial preparation and launch steps](experiment-guide.md#5-prepare-once-convert-then-run) to download, convert, and train the selected model. Use [parameter reference](usage.md) for existing launchers.
