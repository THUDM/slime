"""Deterministic train/rollout alignment gate for Qwen3-30B-A3B.

Drives a full Megatron -> SGLang online-weight-update rollout of Qwen3-30B-A3B
under the deterministic-inference stack (DeepGEMM batch-invariant FP8 forward,
fp32 MoE router, and route-preserving Megatron DeepEP normal dispatch) and
asserts that the training log-probs reproduce the rollout log-probs to better
than ``MAX_TRAIN_ROLLOUT_DIFF`` (default ``1e-6``).

Key differences vs GLM-5.2 gate:
- Standard GQA attention (NOT MLA/DSA) -> attention-backend flash
- All 48 layers are MoE (no dense layers, moe-layer-freq "[1]*48")
- No shared expert, no --spec plugin (standard Megatron MoE)
- moe-router-score-function softmax (not sigmoid), norm_topk_prob=true
- vocab_size 151936
- Model path: HF_MODEL env or /root/Qwen3-30B-A3B-FP8-experts

Environment overrides (all optional; values below are the gate defaults):
* ``MAX_TRAIN_ROLLOUT_DIFF`` -- alignment threshold (default ``1e-6``).
* ``SGLANG_KV_CACHE_DTYPE``  -- ``bfloat16`` (default) or ``fp8_e4m3``.
* ``HF_MODEL``              -- HF checkpoint path (default FP8-experts).
* ``QWEN3_MOE_ALIGNED_SPEC`` -- ``1`` (use aligned attention plugin).
* ``QWEN3_ALIGNED_USE_FUSED_QK_ROPE`` -- ``1``.
* ``SGLANG_MASKED_GEMM_FAST_ACT`` -- ``0`` (overrides alignment_env's ``1``;
  Qwen3 expert dim 768 cannot use the v2 masked quant kernel).
* ``SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK`` -- ``1``.
* ``--rollout-batch-size`` / ``SLIME_E2E_ROLLOUT_BATCH_SIZE`` -- rollout and
  global batch size (default ``8``). CLI wins over the env var.
* ``SLIME_E2E_CONTEXT_PARALLEL_SIZE`` -- Megatron ``--context-parallel-size``
  (default ``1``). ``2`` needs the aligned FA plugin's CP gather path.
* ``MLP_SOCKET_IFNAME``      -- NIC for Ray/NCCL/GLOO/NVSHMEM.
* ``SGLANG_ROOT``           -- deterministic SGLang checkout.
* ``MEGATRON_ROOT``          -- Megatron checkout.
* ``PROMPT_DATA``           -- dataset path.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

NUM_GPUS = 8

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAX_TRAIN_ROLLOUT_DIFF = "9.999e-7"

DEFAULT_HF_MODEL = "/root/Qwen3-30B-A3B-FP8-experts"
DEFAULT_PROMPT_DATA = "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"
DEFAULT_SGLANG_ROOT = "/sgl-workspace/sglang"
DEFAULT_MEGATRON_ROOT = "/root/Megatron-LM"

# Fallback prompt data path used in production run script
_FALLBACK_PROMPT_DATA = "/root/dapo-math-17k/dapo-math-17k.jsonl"

_PREREQ_PROBE = """
import inspect
import deep_gemm
from deep_ep import Buffer
from sglang.srt.server_args import ServerArgs

assert hasattr(deep_gemm, "set_batch_invariant"), "DeepGEMM lacks set_batch_invariant"
assert "align_fp8_quantization" in inspect.signature(Buffer.low_latency_dispatch).parameters, (
    "DeepEP lacks align_fp8_quantization"
)
assert "enable_fp32_moe_router" in ServerArgs.__dataclass_fields__, "SGLang lacks enable_fp32_moe_router"
"""


def _iface_ipv4(ifname: str) -> str | None:
    out = subprocess.run(
        ["ip", "-o", "-4", "addr", "show", ifname], capture_output=True, text=True
    ).stdout.split()
    for i, tok in enumerate(out):
        if tok == "inet" and i + 1 < len(out):
            return out[i + 1].split("/")[0]
    return None


def _pythonpath(sglang_root: str, megatron_root: str) -> str:
    parts = [str(REPO_ROOT), megatron_root, f"{sglang_root}/python"]
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(p for p in parts if p)


def _deterministic_env(
    sglang_root: str,
    megatron_root: str,
    kv_cache_dtype: str,
) -> dict:
    from slime.backends.megatron_utils.alignment.env import alignment_env

    ifname = os.environ.get("MLP_SOCKET_IFNAME")
    env = alignment_env(kv_fp8_qat=kv_cache_dtype == "fp8_e4m3")
    env.update(
        {
            "PYTHONPATH": _pythonpath(sglang_root, megatron_root),
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
            # Qwen3 gate defaults. Must override alignment_env's
            # SGLANG_MASKED_GEMM_FAST_ACT=1 (768 not divisible by 16*128).
            "QWEN3_MOE_ALIGNED_SPEC": os.environ.get("QWEN3_MOE_ALIGNED_SPEC", "1"),
            "QWEN3_ALIGNED_USE_FUSED_QK_ROPE": os.environ.get(
                "QWEN3_ALIGNED_USE_FUSED_QK_ROPE", "1"
            ),
            "SGLANG_MASKED_GEMM_FAST_ACT": os.environ.get(
                "SGLANG_MASKED_GEMM_FAST_ACT", "0"
            ),
            "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": os.environ.get(
                "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK", "1"
            ),
        }
    )
    if ifname:
        env.update(
            {
                "GLOO_SOCKET_IFNAME": ifname,
                "TP_SOCKET_IFNAME": ifname,
                "NCCL_SOCKET_IFNAME": ifname,
                "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": ifname,
            }
        )
    if os.environ.get("NVSHMEM_IBGDA_NIC_HANDLER"):
        env["NVSHMEM_IBGDA_NIC_HANDLER"] = os.environ["NVSHMEM_IBGDA_NIC_HANDLER"]
    return env


def _skip_reason(sglang_root: str, megatron_root: str) -> str | None:
    if not __import__("shutil").which("nvidia-smi"):
        return "nvidia-smi not found (no GPUs)"
    visible = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    n_gpus = len([l for l in visible.stdout.splitlines() if l.strip()])
    if n_gpus < NUM_GPUS:
        return f"needs {NUM_GPUS} GPUs, found {n_gpus}"

    probe_env = {**os.environ, "PYTHONPATH": _pythonpath(sglang_root, megatron_root)}
    probe = subprocess.run(
        [sys.executable, "-c", _PREREQ_PROBE],
        capture_output=True,
        text=True,
        env=probe_env,
    )
    if probe.returncode != 0:
        return f"deterministic stack unavailable: {probe.stdout.strip() or probe.stderr.strip()}"

    if not Path(f"{megatron_root}/megatron/training/tokenizer/tokenizer.py").exists():
        return f"incompatible Megatron root (no tokenizer.tokenizer): {megatron_root}"

    layer_src = Path(f"{megatron_root}/megatron/core/transformer/transformer_layer.py")
    if not (layer_src.exists() and "_use_sglang_fused_residual_rmsnorm" in layer_src.read_text()):
        return f"Megatron root missing megatron-sglang-aligned.patch: {megatron_root}"

    return None


def _resolve_prompt_data(prompt_data: str) -> str:
    """Return first existing path from candidates."""
    if Path(prompt_data).exists():
        return prompt_data
    if Path(_FALLBACK_PROMPT_DATA).exists():
        return _FALLBACK_PROMPT_DATA
    return prompt_data


def _train_args(
    hf_model: str,
    prompt_data: str,
    threshold: str,
    rollout_dump: str,
    kv_cache_dtype: str,
    *,
    rollout_max_response_len: int = 512,
    sglang_layerwise_dump: str | None = None,
    num_layers_override: int | None = None,
    rollout_batch_size: int | None = None,
) -> str:
    """Build the train.py argument string for the Qwen3-30B-A3B alignment gate.

    num_layers_override: if set, override --num-layers (for quick smoke tests).
    """
    num_layers = num_layers_override if num_layers_override is not None else 48
    if rollout_batch_size is None:
        rollout_batch_size = int(os.environ.get("SLIME_E2E_ROLLOUT_BATCH_SIZE", "8"))
    context_parallel_size = int(os.environ.get("SLIME_E2E_CONTEXT_PARALLEL_SIZE", "1"))
    if context_parallel_size < 1:
        raise ValueError(
            f"SLIME_E2E_CONTEXT_PARALLEL_SIZE must be >= 1, got {context_parallel_size}"
        )
    # deepgemm forward layers list: 0..num_layers-1
    deepgemm_layers = " ".join(str(i) for i in range(num_layers))
    # all layers are MoE
    deepgemm_moe_layers = deepgemm_layers

    groups = [
        # placement
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {NUM_GPUS} --colocate "
        "--update-weight-mode full --update-weight-transport nccl "
        "--update-weight-buffer-size 2147483648 --no-check-for-nan-in-loss-and-grad",

        # model: Qwen3-30B-A3B (all MoE layers, standard GQA, no shared expert, no DSA)
        f"--num-layers {num_layers} --hidden-size 2048 --ffn-hidden-size 6144 "
        "--num-attention-heads 32 --num-query-groups 4 --kv-channels 128 "
        f"--moe-layer-freq [1]*{num_layers} "
        "--num-experts 128 --moe-router-topk 8 --moe-grouped-gemm "
        "--moe-ffn-hidden-size 768 "
        "--moe-router-score-function softmax "
        "--moe-router-topk-scaling-factor 1.0 "
        "--moe-aux-loss-coeff 0 --moe-router-dtype fp32 "
        "--vocab-size 151936 "
        "--make-vocab-size-divisible-by 16 "
        "--group-query-attention --qk-layernorm --disable-bias-linear "
        "--swiglu --untie-embeddings-and-output-weights "
        "--position-embedding-type rope --no-position-embedding "
        "--normalization RMSNorm "
        "--rotary-base 1000000 --norm-epsilon 1e-6",

        # checkpoints
        f"--hf-checkpoint {hf_model} --load {hf_model} --ref-load {hf_model}",

        # rollout
        f"--prompt-data {prompt_data} --input-key prompt --label-key label --apply-chat-template "
        f"--rollout-shuffle --rm-type deepscaler --rollout-batch-size {rollout_batch_size} --n-samples-per-prompt 1 "
        f"--global-batch-size {rollout_batch_size} --num-rollout 1 --rollout-max-context-len 4096 "
        f"--rollout-max-response-len {rollout_max_response_len} "
        "--rollout-temperature 1.0 --rollout-top-p 1.0 "
        f"--save-debug-rollout-data {rollout_dump}",

        # optimizer
        "--optimizer adam --lr 2e-6 --lr-warmup-iters 0 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 --no-load-optim --no-save-optim --use-stateless-adam",

        # GRPO
        "--advantage-estimator grpo --kl-loss-coef 0 --kl-loss-type low_var_kl --kl-coef 0 --entropy-coef 0 "
        "--eps-clip 0.2 --eps-clip-high 0.28 --disable-grpo-std-normalization --reset-optimizer-states",

        # parallelism: EP8, TP1 (pure EP, matching production run-qwen3-30B-A3B.sh with EP8)
        f"--tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size {context_parallel_size} --expert-model-parallel-size {NUM_GPUS} --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 --data-pad-size-multiplier 512 "
        "--log-probs-chunk-size 1024",

        # SGLang rollout (deterministic, no DSA, standard flash attention)
        f"--rollout-num-gpus-per-engine {NUM_GPUS} --sglang-server-concurrency 128 "
        "--sglang-mem-fraction-static 0.50 --sglang-enable-dp-attention --sglang-enable-dp-lm-head "
        f"--sglang-ep-size {NUM_GPUS} --sglang-dp-size {NUM_GPUS} --sglang-moe-dp-size 1 "
        f"--sglang-moe-dense-tp-size 1 --sglang-moe-a2a-backend deepep --sglang-deepep-mode {os.environ.get('SLIME_E2E_DEEPEP_MODE', 'low_latency')} "
        "--sglang-moe-runner-backend deep_gemm --sglang-fp8-gemm-runner-backend deep_gemm "
        f"--sglang-page-size 64 --sglang-kv-cache-dtype {kv_cache_dtype} "
        "--sglang-attention-backend fa4 "
        "--sglang-enable-fused-qk-norm-rope "
        "--sglang-chunked-prefill-size 4096 --sglang-context-length 8192 "
        "--sglang-max-prefill-tokens 4096 --sglang-enable-fp32-moe-router "
        "--sglang-enable-deterministic-inference --sglang-disable-prefill-cuda-graph "
        "--sglang-cuda-graph-max-bs-decode 64 --sglang-watchdog-timeout 7200 --sglang-dist-timeout 1800 "
        "--sglang-trust-remote-code",

        # misc + deterministic mode + in-process alignment assertion
        "--attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 --attention-backend flash "
        "--moe-token-dispatcher-type flex --moe-enable-deepep "
        '--train-env-vars {"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","CUDA_LAUNCH_BLOCKING":"1"} '
        "--custom-megatron-before-log-prob-hook-path "
        "slime.backends.megatron_utils.alignment.deepgemm_forward.enable_deepgemm_all_forward "
        "--custom-megatron-before-train-step-hook-path "
        "slime.backends.megatron_utils.alignment.deepgemm_forward.enable_deepgemm_all_forward_before_train_step "
        f"--megatron-deepgemm-forward-layers {deepgemm_layers} "
        f"--megatron-deepgemm-moe-forward-layers {deepgemm_moe_layers} "
        "--deterministic-mode --skip-eval-before-train "
        f"--ci-test --ci-disable-kl-checker --ci-train-rollout-logprob-abs-diff-threshold {threshold}",
    ]

    # Optional: use the SGLang-aligned custom self-attention module. When
    # QWEN3_MOE_ALIGNED_SPEC=1 (default in this gate for the FP32-residual-sum
    # attention alignment work), route through the plugin instead of the
    # default TE-fused SelfAttention.
    if os.environ.get("QWEN3_MOE_ALIGNED_SPEC", "1") == "1":
        # --spec expects two space-separated tokens: <module_location> <function_name>
        groups.append(
            "--spec slime_plugins.models.qwen3_moe_aligned get_qwen3_moe_aligned_spec"
        )

    if os.environ.get("SLIME_E2E_DISABLE_DECODE_CG", "0") == "1":
        groups.append("--sglang-disable-decode-cuda-graph")

    if sglang_layerwise_dump is not None:
        layer_list = " ".join(str(i) for i in range(num_layers))
        groups.append(
            f"--sglang-debug-tensor-dump-output-folder {sglang_layerwise_dump} "
            f"--sglang-debug-tensor-dump-layers {layer_list}"
        )

    return " ".join(groups)


def run_gate(
    *,
    layerwise_zero: bool = False,
    rollout_max_response_len: int = 32,
    num_layers_override: int | None = None,
    rollout_batch_size: int | None = None,
) -> None:
    sglang_root = os.environ.get("SGLANG_ROOT", DEFAULT_SGLANG_ROOT)
    megatron_root = os.environ.get("MEGATRON_ROOT", DEFAULT_MEGATRON_ROOT)
    hf_model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
    prompt_data = _resolve_prompt_data(os.environ.get("PROMPT_DATA", DEFAULT_PROMPT_DATA))
    threshold = os.environ.get("MAX_TRAIN_ROLLOUT_DIFF", DEFAULT_MAX_TRAIN_ROLLOUT_DIFF)
    kv_cache_dtype = os.environ.get("SGLANG_KV_CACHE_DTYPE", "bfloat16")
    if kv_cache_dtype not in {"bfloat16", "fp8_e4m3"}:
        raise ValueError(
            f"SGLANG_KV_CACHE_DTYPE must be bfloat16 or fp8_e4m3, got {kv_cache_dtype!r}"
        )

    reason = _skip_reason(sglang_root, megatron_root)
    if reason is not None:
        msg = f"Qwen3-30B-A3B deterministic gate skipped: {reason}"
        print(msg, flush=True)
        pytest.skip(msg)

    if not Path(hf_model, "config.json").exists():
        raise FileNotFoundError(f"HF checkpoint missing: {hf_model}/config.json")
    if not Path(prompt_data).exists():
        raise FileNotFoundError(f"Prompt data missing: {prompt_data}")

    master_addr = "127.0.0.1"
    ifname = os.environ.get("MLP_SOCKET_IFNAME")
    if ifname and (ip := _iface_ipv4(ifname)):
        master_addr = ip
    master_port = os.environ.get("MASTER_PORT", "29500")

    env = {
        **os.environ,
        **_deterministic_env(sglang_root, megatron_root, kv_cache_dtype),
    }
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(proxy_var, None)
    env["MASTER_ADDR"] = master_addr
    env["RAY_ADDRESS"] = f"{master_addr}:{master_port}"

    num_layers = num_layers_override if num_layers_override is not None else 48
    if rollout_batch_size is None:
        rollout_batch_size = int(os.environ.get("SLIME_E2E_ROLLOUT_BATCH_SIZE", "8"))
    gate_name = "layerwise-zero" if layerwise_zero else "train/rollout"
    print(
        f"Running Qwen3-30B-A3B deterministic {gate_name} gate "
        f"(layers={num_layers}, samples={rollout_batch_size}, "
        f"logprob limit={float(threshold):g}) ...",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="qwen3_30b_gate_") as tmp:
        rollout_dump = os.path.join(tmp, "rollout_data", "{rollout_id}.pt")
        megatron_layerwise_dump = os.path.join(tmp, "megatron_layerwise")
        sglang_layerwise_dump = os.path.join(tmp, "sglang_layerwise")

        if layerwise_zero:
            env.update(
                {
                    "SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR": megatron_layerwise_dump,
                    "SGLANG_TENSOR_DUMP_LAYER_OUTPUTS_ONLY": "1",
                    "SGLANG_TENSOR_DUMP_CHUNK_SIZE": "64",
                }
            )

        argv = _train_args(
            hf_model,
            prompt_data,
            threshold,
            rollout_dump,
            kv_cache_dtype,
            rollout_max_response_len=rollout_max_response_len,
            sglang_layerwise_dump=(sglang_layerwise_dump if layerwise_zero else None),
            num_layers_override=num_layers_override,
            rollout_batch_size=rollout_batch_size,
        ).split()

        _run(["pkill", "-9", "sglang"], check=False)
        _run(["ray", "stop", "--force"], check=False, env=env)
        try:
            _run(
                [
                    "ray",
                    "start",
                    "--head",
                    "--node-ip-address",
                    master_addr,
                    "--num-gpus",
                    str(NUM_GPUS),
                    "--port",
                    master_port,
                    "--disable-usage-stats",
                    "--include-dashboard=false",
                ],
                env=env,
            )
            code, _out = _run(
                [sys.executable, "-u", "train.py", *argv],
                env=env,
                cwd=str(REPO_ROOT),
                stream=True,
            )
            assert code == 0, (
                f"train.py exited {code} (train/rollout bound {float(threshold):g} likely breached)"
            )
            if layerwise_zero:
                layer_ids = [str(i) for i in range(num_layers)]
                _run(
                    [
                        sys.executable,
                        "-m",
                        "slime.utils.compare_glm52_layerwise",
                        "--megatron-dir",
                        megatron_layerwise_dump,
                        "--sglang-dir",
                        sglang_layerwise_dump,
                        "--layers",
                        *layer_ids,
                        "--max-hidden-diff",
                        "0",
                    ],
                    env=env,
                    cwd=str(REPO_ROOT),
                    stream=True,
                )
        finally:
            _run(["ray", "stop", "--force"], check=False, env=env)

    print(
        f"Qwen3-30B-A3B deterministic {gate_name} gate PASSED "
        f"(layers={num_layers}, logprob limit={float(threshold):g})",
        flush=True,
    )


def _run(cmd, env=None, cwd=None, check=True, stream=False):
    if not stream:
        r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"{cmd} failed ({r.returncode}): {r.stdout}\n{r.stderr}")
        return r.returncode, (r.stdout or "")
    proc = subprocess.Popen(
        cmd, env=env, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    lines = []
    for line in proc.stdout:
        lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    return proc.returncode, "".join(lines)


def test_qwen3_30b_a3b_deterministic_train_rollout_alignment(request):
    run_gate(rollout_batch_size=request.config.getoption("rollout_batch_size"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-rs"]))
