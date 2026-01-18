# Reproducibility

Reproducibility is a bedrock of scientific progress. 通过结合 SGLang 提供的 [确定性推理](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) 和 Megatron-LM 的确定性模式，slime 可以提供完全确定性（bitwise）的实验复现能力。

为了开启确定性训练，你需要通过 `pip uninstall flash_attn_3 -y` 卸载 flash attention 3，并设置：

```bash
  # sglang config
  --sglang-enable-deterministic-inference
  --sglang-attention-backend flashinfer

  # megatron config
  --deterministic-mode
```

以及设置如下环境变量：

```bash
     "env_vars": {
        ...,
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
```

我们提供了一个完全确定性的，用 Qwen2.5 0.5B 训练 GSM8K 的脚本。

可以用如下脚本初始化训练数据和 ckpt：

```bash
# download
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# convert ckpt
cd slime/
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM/ python \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
   --save /root/Qwen2.5-0.5B-Instruct_torch_dist/
```

可以使用如下脚本进行训练：

```bash
bash script/run-qwen2.5-0.5B-reproducibility.sh
```

这个 PR 中记录了 wandb 的截图 [pull#370](https://github.com/THUDM/slime/pull/370).
