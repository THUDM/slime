# Slime NPU Patch Installation Guide

This guide provides instructions for installing Slime with NPU support, including all required dependencies and patches.

## Component Version Mapping

| Component       | Version/Commit                           | Source                                                                                                              |
| --------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Slime           | v0.2.2                                   | [GitHub](https://github.com/THUDM/slime/tree/v0.2.2)                                                                |
| SGLang          | dce8b0606c06d3a191a24c7b8cbe8e238ab316c9 | [GitHub](https://github.com/sgl-project/sglang/tree/sglang-slime)                                             |
| SGL Kernel NPU  | 2026.02.01                               | [GitHub](https://github.com/sgl-project/sgl-kernel-npu/releases/tag/2026.02.01)                                     |
| Megatron-Bridge | 35b4ebfc486fb15dcc0273ceea804c3606be948a | [GitHub](https://github.com/fzyzcjy/Megatron-Bridge)                                                                |
| Megatron-LM     | 3714d81d418c9f1bca4594fc35f9e8289f652862 | [GitHub](https://github.com/NVIDIA/Megatron-LM)                                                                     |
| MindSpeed       | fc63de5c48426dd019c3b3f39e65f5bdf56e4086 | [GitCode](https://gitcode.com/Ascend/MindSpeed)                                                                     |
| HDK             | 25.3.RC1                                 | [Ascend](https://www.hiascend.com/hardware/firmware-drivers/commercial?product=7\&model=33)                         |
| CANN            | 8.5.0                                    | [Ascend](https://www.hiascend.com/developer/download/community/result?module=cann\&cann=8.5.0\&product=7\&model=33) |

## Preparing the Running Environment

### Python Version

Only `python==3.11` is supported currently.

```shell
conda create -n slime_release python=3.11
conda activate slime_release
```

### Working Directory Setup

```shell
mkdir <WORKSPACE> && cd <WORKSPACE>
```

### CANN Environment

Prior to start work with Slime on Ascend you need to install CANN Toolkit, Kernels operator package and NNAL version 8.5.0, check the [installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns\&InstallType=local\&OS=openEuler\&Software=cannToolKit)

```shell
source <CANN_PATH>/ascend-toolkit/set_env.sh
source <CANN_PATH>/nnal/atb/set_env.sh
```

### PyTorch and PyTorch NPU

```shell
pip install torch-npu==2.8.0
```

## Installing Dependencies

### SGLang

```shell
cd <WORKSPACE>
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout dce8b0606c06d3a191a24c7b8cbe8e238ab316c9
mv python/pyproject.toml python/pyproject.toml.backup
mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[srt_npu]"
pip install torch-npu==2.8.0
```

### SGL Kernel NPU and Torch Memory Saver

Download `sgl-kernel-npu-2026.02.01-torch2.8.0-py311-cann8.5.0-a3-aarch64.zip` from the release link, then install:

```shell
pip install sgl_kernel_npu-2026.2.1-cp311-cp311-linux_aarch64.whl
pip install torch_memory_saver-0.0.8-cp311-cp311-linux_aarch64.whl
```

### Megatron-Bridge

```shell
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

cd <WORKSPACE>
git clone https://github.com/fzyzcjy/Megatron-Bridge.git -b dev_rl
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
```

### Megatron-LM

```shell
cd <WORKSPACE>
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
  cd Megatron-LM/ && git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862 && \
  pip install -e .
```

### MindSpeed

```shell
cd <WORKSPACE>
git clone https://gitcode.com/Ascend/MindSpeed.git && \
  cd MindSpeed/ && git checkout fc63de5c48426dd019c3b3f39e65f5bdf56e4086 && \
  pip install -e .
```

### Slime

```shell
cd <WORKSPACE>
git clone https://github.com/ascend-slime/slime.git && cd slime
cp -r docker/npu_patch ../npu_patch
git checkout v0.2.2
pip install -e .
```

## Applying Patches

```shell
cd <WORKSPACE>/slime
git apply ../npu_patch/slime.patch

cd <WORKSPACE>/sglang
git apply ../slime/docker/patch/v0.5.7/sglang.patch
git apply ../npu_patch/sglang.patch

cd <WORKSPACE>/Megatron-LM
git apply ../slime/docker/patch/v0.5.7/megatron.patch
git apply ../npu_patch/megatron.patch

cd <WORKSPACE>/Megatron-Bridge
git apply ../npu_patch/megatron-bridge.patch

cd <WORKSPACE>/MindSpeed
git apply ../npu_patch/mindspeed.patch
```

## Additional Dependencies

```shell
cd <WORKSPACE>/slime
pip install triton-ascend
pip install torch-npu==2.8.0
pip install torchvision==0.23.0
pip install numpy==1.26.0
```

## Running the Training

### Configuration

Modify the paths in the following files according to your environment (note to use your CANN version):

**Common (both GRPO and PPO):**

- `slime/utils/external_utils/command_utils.py`

**GRPO:**

- `examples/geo3k_vlm_multi_turn/run_grpo_npu.sh`
- `examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn_grpo_npu.py`

**PPO:**

- `examples/geo3k_vlm_multi_turn/run_ppo_npu.sh`
- `examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn_ppo_npu.py`

### Dataset

Download the dataset from [HuggingFace](https://huggingface.co/datasets/VeraIsHere/geo3k_imgurl_processed) following the instructions in the script directory.

### Execute Training

```shell
cd <WORKSPACE>/slime
# GRPO
bash examples/geo3k_vlm_multi_turn/run_grpo_npu.sh
# PPO
bash examples/geo3k_vlm_multi_turn/run_ppo_npu.sh
```

To save logs and display them simultaneously:

```shell
# GRPO
bash examples/geo3k_vlm_multi_turn/run_grpo_npu.sh 2>&1 | tee -a <LOG_FILE>
# PPO
bash examples/geo3k_vlm_multi_turn/run_ppo_npu.sh 2>&1 | tee -a <LOG_FILE>
```

## Placeholders Reference

| Placeholder   | Description                          | Example               |
| ------------- | ------------------------------------ | --------------------- |
| `<WORKSPACE>` | Root directory for all installations | `/root/slime-release` |
| `<CANN_PATH>` | Path to CANN installation directory  | `/usr/local/ascend`   |
| `<LOG_FILE>`  | Path to log file for training output | `training.log`        |

