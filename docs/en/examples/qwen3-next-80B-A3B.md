# Training Qwen3-Next-80B-A3B on 8xH100

## Environment Setup

The environment setup, model download, data preparation, and checkpoint conversion are the same as for the Qwen3-4B model. Please refer to [Example: Qwen3-4B](./qwen3-4B.md), replacing the Qwen3-4B references with Qwen3-next-80B-A3B-Instruct.

You can use the following method to convert the HuggingFace checkpoint to torch_dist format:

```bash
export BASE_FOLDER=./models/
# Download model weights (Qwen3-Next-80B-A3B-Thinking)
hf download Qwen/Qwen3-Next-80B-A3B-Thinking --local-dir ${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking
```

```shell
cd slime/
pip install -e .

# (for acceleration)
cd .. # and find a proper folder
git clone https://github.com/fla-org/flash-linear-attention
cd flash-linear-attention
git checkout 9714c595
pip install -e .

wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install ./causal_conv1d-1.5.4+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## [Optional] Fix a bug in triton compilation on Blackwell (sm100)

See discussion here https://github.com/triton-lang/triton/issues/8695
and https://github.com/fla-org/flash-linear-attention/issues/638

We need to apply a patch to fix the bug.
Go to the flash-linear-attention folder you just installed, and apply the following patch:

```diff
diff --git a/fla/ops/gated_delta_rule/wy_fast.py b/fla/ops/gated_delta_rule/wy_fast.py
index c5119dcf..838f5e4e 100644
--- a/fla/ops/gated_delta_rule/wy_fast.py
+++ b/fla/ops/gated_delta_rule/wy_fast.py
@@ -198,7 +198,14 @@ def prepare_wy_repr_bwd_kernel(
         b_A += tl.dot(b_kb, tl.trans(b_k))
         b_dkb = tl.dot(b_dA, b_k)
         b_db += tl.sum(b_dkb * b_k, 1)
-        b_dk += tl.dot(tl.trans(b_dA), b_kb)
+        b_dk += tl.inline_asm_elementwise(
+            asm="mov.f32 $0, $1;",
+            constraints="=r,r",
+            args=[tl.dot(tl.trans(b_dA), b_kb)],
+            dtype=tl.float32,
+            is_pure=True,
+            pack=1,
+        )
         b_dk += b_dkb * b_b[:, None]
         tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
     tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

```

Save it as `patch.diff` (Please remember to copy the last empty line to the file!) and run `git apply patch.diff`

## Training (Megatron)

**Note: Blackwell is currently not supported**

Convert model weights:

```bash
source scripts/models/qwen3-next-80B-A3B.sh
PYTHONPATH=/root/Megatron-LM/ torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-Next-80B-A3B-Thinking/ \
   --save /root/Qwen3-Next-80B-A3B-Thinking_torch_dist/
```

Single node with 8 GPUs:

```bash
cd /root/slime
export BASE_FOLDER=/root
export MASTER_ADDR=127.0.0.1
bash scripts/run-qwen3-next-80B-A3B-8gpus.sh
```

If you run out of memory, consider disabling `--accumulate-allreduce-grads-in-fp32` and enabling `--grad-reduce-in-bf16`.

Multi-node (4x8):

```bash
cd /root/slime
export BASE_FOLDER=/root
export MASTER_ADDR=your_master_addr
bash scripts/run-qwen3-next-80B-A3B.sh
```

## Training (FSDP)

```bash
export BASE_FOLDER=./models/
export MASTER_ADDR=127.0.0.1

bash scripts/run-qwen3-next-80B-A3B-fsdp.sh
```
