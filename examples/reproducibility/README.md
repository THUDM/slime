# Reproducibility

Reproducibility is a bedrock of scientific progress. By combining the [deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) of SGLang and the deterministic mode of Megatron-LM, slime supports bitwise experiment reproduction.

To enable deterministic training, you need to set:
```bash
  # sglang config
  --sglang-enable-deterministic-inference
  --sglang-attention-backend flashinfer

  # megatron config
  --deterministic-mode
```

And set the following environment variables:

```bash
     "env_vars": {
        ...,
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
```

We also need to set `--use-slime-router` until the pypi whl of sglang-router updates.

Here we provide the script to do RL training on Qwen2.5 0.5B model and GSM8K dataset with full deterministic.

For data and checkpoint preparation, please run:

```bash
# download
huggingface-cli download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# convert ckpt
cd slime/
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM/ python \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
   --save /root/Qwen2.5-0.5B-Instruct_torch_dist/
```

And to run training,

```bash
bash examples/reproducibility/run-qwen2.5-0.5B-gsm8k.sh
```

For screen shots of the wandb, please refer to [pull#370](https://github.com/THUDM/slime/pull/370).
