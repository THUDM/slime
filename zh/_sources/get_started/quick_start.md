# 运行环境准备

从 {ref}`交互 Quick Start <lab>` 开始。本页准备所有生成配方共用的环境；[tutorial](experiment-guide.md)讲解完整 RL 流程。

## 基础环境搭建

建议使用预置的 Docker 镜像，获得相互兼容的 SGLang、Megatron 和补丁环境。

### 硬件与镜像选择

| 硬件 | 镜像 | CUDA |
| --- | --- | --- |
| H100/H200 | `slimerl/slime:latest` 或 `slimerl/slime:latest-cu129` | 12.9 |
| B200/B300（Blackwell，x86） | `slimerl/slime:latest-cu130` | 13.0 |

默认 `latest` 标签对应 CUDA 12 镜像。使用 Blackwell 时，请将下方命令中的镜像替换为 `slimerl/slime:latest-cu130`。两种镜像使用不同的 kernel 和 Transformer Engine 构建方式；依赖版本和构建选项见 [Docker 指南](https://github.com/THUDM/slime/blob/main/docker/README.md)。

GPU CI 主要覆盖 H100/H200。使用其他平台时，请先核对具体训练配方的硬件和并行配置要求。

不方便使用 Docker 时，可使用{ref}`下方的 conda 方案 <conda>`。AMD 支持见 [AMD 使用教程](../platform_support/amd_tutorial.md)。

### 拉取并启动 Docker 容器

在 Linux x86_64 GPU 主机上准备 Git、Docker、兼容的 NVIDIA 驱动，并配置好 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。在**宿主机上**执行以下命令，拉取镜像、克隆 slime 并启动容器：

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

### 安装 slime

镜像已经包含 slime 及所需依赖；挂载后，镜像内的代码目录被宿主机 checkout 替代。在**容器内**执行以下命令，安装该 checkout、保留镜像中固定的依赖版本，并检查模块导入和 GPU 可见性：

```bash
pip install -e . --no-deps
python -c "import slime, sglang, megatron.core; import torch; print('GPUs:', torch.cuda.device_count())"
```

输出的 GPU 数量应与容器可见设备一致。将配置器导出的 `experiment.sh` 保存到宿主机的 `slime` 目录后，即可在容器内的 `/root/slime/experiment.sh` 访问它。容器内 `/data` 下的模型、数据及 checkpoint 会保存在宿主机旁边的 `slime-data` 目录，退出容器后仍然保留。再次进入环境时，在同一 checkout 下重新执行 `docker run` 命令，并在新容器内执行上述 editable install。

(conda)=
## Conda / micromamba

[build_conda.sh](https://github.com/THUDM/slime/blob/main/build_conda.sh) 会安装 micromamba，创建使用 Python 3.12 和 CUDA 12.9 的 `slime` 环境，构建 Megatron、SGLang 及原生依赖，安装 slime 并应用仓库的配套补丁。首次构建会比启动预置镜像更久。

适用于 Linux x86_64 NVIDIA 机器，需要 Bash、Git、curl、patch、C/C++ 编译工具及兼容的 GPU 驱动。当前脚本会写入 `/root`（包括 `/root/.cargo`），需要相应权限；以下命令使用脚本默认的 `/root` 源码目录。B200/B300 请使用 CUDA 13 Docker 镜像。

在 Bash 中执行：

```bash
git clone https://github.com/THUDM/slime.git
cd slime
BASE_DIR=/root SHELL=/bin/bash bash build_conda.sh
```

构建成功后，在当前终端中激活环境。每个新 Bash 会话中都需重新执行以下激活命令：

```bash
source ~/.bashrc
micromamba activate slime
export CUDA_HOME="$CONDA_PREFIX"
python -c "import slime, sglang, megatron.core; import torch; print('GPUs:', torch.cuda.device_count())"
```

继续在 slime 仓库根目录运行生成的脚本。在配置器的「文件」步骤，将模型、数据和输出路径设为本机可写目录；按上述安装方式，Megatron 路径为 `/root/Megatron-LM`。

## 挂载实验路径

上述 Docker 命令已经将持久化数据目录挂载到 `/data`，与配置器默认值一致。使用已有数据或共享存储时，可将该挂载替换为 `-v /shared/data:/data`，并在向导中填写容器内可见路径。所有 Ray 节点以相同路径访问仓库和生成配置目录，安装相同依赖，并配置节点之间可达的网络；上述单机 Docker 命令不会自动搭好多机网络。

继续阅读 [tutorial 的准备与启动步骤](experiment-guide.md)，完成所选模型的下载、转换和训练。已有 launcher 的参数说明见[使用参考](usage.md)。
