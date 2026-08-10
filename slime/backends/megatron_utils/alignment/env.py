"""Deterministic train/rollout alignment environment.

Centralizes the numerical-alignment environment variables that both train
(Megatron) and rollout (SGLang) actors must share for GLM-5 train/rollout
log-prob alignment. Launchers (and the 6-layer gate test) merge this with
their own connectivity settings (``PYTHONPATH``, ``MASTER_ADDR``, NIC names,
proxy, IBGDA handler), which are cluster-specific and intentionally not here.
"""

from __future__ import annotations

from pathlib import Path

# slime/backends/sglang_utils/jit_kernels — the slime-hosted custom SGLang
# kernels (e.g. glm5_router_gemm) that sglang's JIT loader try-cache compiles.
_JIT_KERNELS_DIR = Path(__file__).resolve().parents[2] / "sglang_utils" / "jit_kernels"


def alignment_env(*, kv_fp8_qat: bool = False) -> dict[str, str]:
    """Return the shared deterministic-alignment env vars.

    ``kv_fp8_qat`` enables the FP8-E4M3 KV-cache QAT path (bf16 KV when False).
    """
    return {
        # Deterministic collectives / matmul.
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_ALGO": "^NVLS",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCH_COMPILE_DISABLE": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "TE_DISABLE_FA3": "TRUE",
        "NVSHMEM_DISABLE_NCCL": "1",
        # DeepGEMM batch-invariant FP8 forward.
        "SGLANG_DEEPGEMM_BATCH_INVARIANT": "1",
        "SGLANG_DEEPGEMM_PAD_EXPERT_M": "1",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGLANG_JIT_KERNEL_EXTRA_PATH": str(_JIT_KERNELS_DIR),
        "SGLANG_MASKED_GEMM_FAST_ACT": "1",
        # DeepEP low-latency dispatch + DSA indexer.
        "SGLANG_DEEPEP_LL_PREFILL_STAGING": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
        "SGLANG_DSA_FUSE_TOPK": "0",
        "SGLANG_DISABLE_DSA_INDEXER_FUSION": "1",
        "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
        "INDEXER_ROPE_NEOX_STYLE": "0",
        # Megatron train side borrows SGLang's aligned kernels.
        "MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS": "1",
        "MEGATRON_USE_SGLANG_FP8_INDEXER": "1",
        "MEGATRON_USE_SGLANG_ROUTER_GEMM": "1",
        "MEGATRON_USE_SGLANG_ROPE": "1",
        "MEGATRON_USE_SGLANG_SPARSE_MLA": "1",
        # DSA KV cache dtype.
        "DSA_KV_FP8_QAT": "1" if kv_fp8_qat else "0",
        "DSA_KV_FP8_QAT_BLOCK_SIZE": "128",
    }
