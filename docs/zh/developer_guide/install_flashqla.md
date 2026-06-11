# 安装 FlashQLA

FlashQLA 是 Qwen GDN kernel 的可选运行后端。安装 FlashQLA 后，仍需要在训练命令中显式加入：

```bash
--qwen-gdn-backend flashqla
```

如果不传该参数，Qwen GDN 仍使用默认的 FLA 后端。

## 环境要求

使用 `--qwen-gdn-backend flashqla` 前，请确认训练节点满足：

- PyTorch 2.8 或更新版本。
- CUDA 12.8 或更新版本。
- NVIDIA SM90 或更新架构 GPU。
- 所有训练节点都安装了同一套 FlashQLA Python 包。

## Conda / 本地环境

`build_conda.sh` 会默认安装 FlashQLA：

```bash
bash build_conda.sh
```

## Docker 镜像

标准 CUDA Docker 镜像会默认安装 FlashQLA：

```bash
docker build \
  -f docker/Dockerfile \
  -t slime:flashqla .
```

## GB10 镜像

`docker/Dockerfile.gb10` 默认不安装 FlashQLA。GB10 环境需要自行验证 FlashQLA 编译和运行行为；如需实验性安装，可显式传入：

```bash
docker build \
  --build-arg INSTALL_FLASHQLA=1 \
  -f docker/Dockerfile.gb10 \
  -t slime:gb10-flashqla .
```
