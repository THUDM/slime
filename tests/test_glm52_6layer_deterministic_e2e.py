"""Deterministic train/rollout alignment gate for a 6-layer GLM-5 model.

Drives a full Megatron -> SGLang online-weight-update rollout of a 6-layer
GLM-5.2 structure model under the deterministic-inference stack (DSA attention,
DeepGEMM batch-invariant FP8 forward, fp32 MoE router, and route-preserving
Megatron DeepEP normal dispatch) and asserts that the training log-probs reproduce the rollout log-probs
to better than ``MAX_TRAIN_ROLLOUT_DIFF`` (default ``1e-6``). The bound is
enforced in-process via ``--ci-test --ci-train-rollout-logprob-abs-diff-threshold``.

This test is self-contained: the model / rollout / deterministic config lives
here (not in any external experiment directory). It needs the deterministic
SGLang / DeepGEMM / DeepEP build plus the 6-layer GLM-5.2 checkpoints, so it
self-skips (exit 0) whenever any prerequisite is missing -- a no-op on runners
that lack the bespoke stack, a real gate wherever it is installed.

Environment overrides (all optional):

* ``MAX_TRAIN_ROLLOUT_DIFF`` -- alignment threshold (default ``1e-6``).
* ``SGLANG_KV_CACHE_DTYPE``  -- ``fp8_e4m3`` (default) or ``bfloat16``.
* ``MLP_SOCKET_IFNAME``      -- NIC for Ray/NCCL/GLOO/NVSHMEM (single-node run).
* ``SGLANG_ROOT``           -- deterministic SGLang checkout (default
  ``/mnt/zhuzilin/github-sglang``); its ``python`` dir is prepended to PYTHONPATH.
* ``MEGATRON_ROOT``          -- Megatron checkout (default ``/root/Megatron-LM``).
* ``HF_MODEL`` / ``NATIVE_CKPT`` / ``PROMPT_DATA`` -- checkpoints / dataset.
* ``NVSHMEM_IBGDA_NIC_HANDLER`` -- pass through for Blackwell RoCE fabrics.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Single-node EP8 gate; run-ci-changed reads this top-level constant.
NUM_GPUS = 8

REPO_ROOT = Path(__file__).resolve().parents[1]

# "train rollout diff below x e-6": require the deterministic run to land below
# 1e-6 (the H100 reference lands ~2e-7, leaving a comfortable margin).
DEFAULT_MAX_TRAIN_ROLLOUT_DIFF = "9.999e-7"

# Reference locations on the alignment cluster; every one is overridable via env.
DEFAULT_HF_MODEL = "/mnt/zhuzilin/slime/train_rollout_diff_exp/artifacts/checkpoints/glm52_6layer_hf_fp8"
DEFAULT_NATIVE_CKPT = (
    "/mnt/o1_checkpoints/chat/glm-4.6/700b-moe/"
    "700b-sft-dsa-indexshare4-512k-260525-glm5p2-v3-rev1-fastdecay/iter_0000500"
)
DEFAULT_PROMPT_DATA = "/mnt/o1_alicloud/personal/yc/data/RL/RL_train_prompt/hle-v6_critpt-v2.jsonl"
DEFAULT_SGLANG_ROOT = "/mnt/zhuzilin/github-sglang"
DEFAULT_MEGATRON_ROOT = "/root/Megatron-LM"

# Probe (run under the gate PYTHONPATH) for the deterministic-inference stack.
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
    out = subprocess.run(["ip", "-o", "-4", "addr", "show", ifname], capture_output=True, text=True).stdout.split()
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
    """Shared alignment env + this launch's connectivity settings."""
    from slime.backends.megatron_utils.alignment.env import alignment_env

    ifname = os.environ.get("MLP_SOCKET_IFNAME")
    env = alignment_env(
        kv_fp8_qat=kv_cache_dtype == "fp8_e4m3",
    )
    env.update(
        {
            "PYTHONPATH": _pythonpath(sglang_root, megatron_root),
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
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
    # Opt-in CPU-assisted IBGDA for DeepEP low-latency on Blackwell RoCE fabrics.
    if os.environ.get("NVSHMEM_IBGDA_NIC_HANDLER"):
        env["NVSHMEM_IBGDA_NIC_HANDLER"] = os.environ["NVSHMEM_IBGDA_NIC_HANDLER"]
    return env


def _skip_reason(sglang_root, megatron_root, hf_model, native_ckpt, prompt_data) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found (no GPUs)"
    visible = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    )
    n_gpus = len([line for line in visible.stdout.splitlines() if line.strip()])
    if n_gpus < NUM_GPUS:
        return f"needs {NUM_GPUS} GPUs, found {n_gpus}"

    probe_env = {**os.environ, "PYTHONPATH": _pythonpath(sglang_root, megatron_root)}
    probe = subprocess.run([sys.executable, "-c", _PREREQ_PROBE], capture_output=True, text=True, env=probe_env)
    if probe.returncode != 0:
        return f"deterministic stack unavailable: {probe.stdout.strip() or probe.stderr.strip()}"

    if not Path(f"{megatron_root}/megatron/training/tokenizer/tokenizer.py").exists():
        return f"incompatible Megatron root (no tokenizer.tokenizer): {megatron_root}"
    # The FP32 residual/RMSNorm boundary (docker/patch/latest/megatron-sglang-aligned.patch)
    # is required for train/rollout alignment; without it the gate diverges (~1e-2)
    # instead of failing to launch, so skip rather than misreport.
    layer_src = Path(f"{megatron_root}/megatron/core/transformer/transformer_layer.py")
    if not (layer_src.exists() and "_use_sglang_fused_residual_rmsnorm" in layer_src.read_text()):
        return f"Megatron root missing megatron-sglang-aligned.patch: {megatron_root}"
    for label, path in (
        ("HF checkpoint", f"{hf_model}/config.json"),
        ("native checkpoint", native_ckpt),
        ("prompt data", prompt_data),
    ):
        if not Path(path).exists():
            return f"{label} missing: {path}"
    return None


def _native_checkpoint_view(native_ckpt: str, root: str) -> str:
    """Megatron wants a checkpoint root with a tracker file; the production ckpt
    ships as an ``iter_*`` leaf. Expose a non-copying root view."""
    leaf = os.path.basename(native_ckpt.rstrip("/"))
    if not (leaf.startswith("iter_") and leaf[5:].isdigit()):
        return native_ckpt
    view = os.path.join(root, "native_checkpoint_view")
    os.makedirs(view, exist_ok=True)
    with open(os.path.join(view, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(f"{int(leaf[5:])}\n")
    link = os.path.join(view, leaf)
    if not os.path.lexists(link):
        os.symlink(native_ckpt, link)
    return view


def _train_args(
    hf_model,
    ref_load,
    prompt_data,
    threshold,
    rollout_dump,
    kv_cache_dtype,
    *,
    rollout_max_response_len=4096,
    sglang_layerwise_dump=None,
) -> str:
    groups = [
        # placement (single-node, colocated)
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {NUM_GPUS} --colocate "
        "--update-weight-mode full --update-weight-transport nccl "
        "--update-weight-buffer-size 2147483648 --no-check-for-nan-in-loss-and-grad",
        # model (6-layer GLM-5.2 structure: 3 dense + 3 MoE, MLA + DSA)
        "--spec slime_plugins.models.glm5.glm5 get_glm5_spec "
        "--moe-layer-freq [0]*3+[1]*3 --num-experts 256 --moe-shared-expert-intermediate-size 2048 "
        "--moe-router-topk 8 --moe-grouped-gemm --moe-ffn-hidden-size 2048 "
        "--moe-router-score-function sigmoid --moe-router-pre-softmax --moe-router-enable-expert-bias "
        "--moe-router-bias-update-rate 0 --moe-router-load-balancing-type seq_aux_loss "
        "--moe-router-topk-scaling-factor 2.5 --moe-aux-loss-coeff 0 --moe-router-dtype fp32 "
        "--make-vocab-size-divisible-by 16 --num-layers 6 --hidden-size 6144 --ffn-hidden-size 12288 "
        "--num-attention-heads 64 --disable-bias-linear --swiglu --untie-embeddings-and-output-weights "
        "--position-embedding-type rope --no-position-embedding --normalization RMSNorm --qk-layernorm "
        "--multi-latent-attention --q-lora-rank 2048 --kv-lora-rank 512 --qk-head-dim 192 --v-head-dim 256 "
        "--kv-channels 192 --qk-pos-emb-head-dim 64 --vocab-size 154880 --rotary-base 8000000 "
        "--enable-experimental",
        # checkpoints
        f"--hf-checkpoint {hf_model} --ref-load {ref_load}",
        # rollout
        f"--prompt-data {prompt_data} --input-key input_messages --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type deepscaler --rollout-batch-size 8 --n-samples-per-prompt 1 "
        "--global-batch-size 8 --num-rollout 1 --rollout-max-context-len 4096 "
        f"--rollout-max-response-len {rollout_max_response_len} "
        "--rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-stop-token-ids 154820 154827 154829 "
        f"--save-debug-rollout-data {rollout_dump}",
        # optimizer
        "--optimizer adam --lr 2e-6 --lr-warmup-iters 0 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 --no-load-optim --no-save-optim --optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer",
        # GRPO + TIS (icepop)
        "--advantage-estimator grpo --kl-loss-coef 0 --kl-loss-type low_var_kl --kl-coef 0 --entropy-coef 0 "
        "--eps-clip 0.2 --eps-clip-high 0.28 --use-tis "
        "--custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function "
        "--tis-clip-low 0.5 --tis-clip 2.0 --disable-grpo-std-normalization --reset-optimizer-states",
        # parallelism / perf (pure-EP, recompute, dynamic batch)
        f"--tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {NUM_GPUS} --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 --data-pad-size-multiplier 512 "
        # This is deliberately a full-main-model-parameter gate. In particular,
        # routed and shared experts must execute backward through Megatron
        # DeepEP; only the auxiliary DSA indexer remains frozen below.
        "--log-probs-chunk-size 1024",
        # SGLang rollout (deterministic DSA + DeepEP low-latency + DeepGEMM)
        f"--rollout-num-gpus-per-engine {NUM_GPUS} --sglang-server-concurrency 128 "
        "--sglang-mem-fraction-static 0.70 --sglang-enable-dp-attention --sglang-enable-dp-lm-head "
        f"--sglang-ep-size {NUM_GPUS} --sglang-dp-size {NUM_GPUS} --sglang-moe-dp-size 1 "
        "--sglang-moe-dense-tp-size 1 --sglang-moe-a2a-backend deepep --sglang-deepep-mode low_latency "
        "--sglang-moe-runner-backend deep_gemm --sglang-fp8-gemm-runner-backend deep_gemm "
        f"--sglang-page-size 64 --sglang-kv-cache-dtype {kv_cache_dtype} --sglang-attention-backend dsa "
        "--sglang-dsa-prefill-backend flashmla_sparse --sglang-dsa-decode-backend flashmla_sparse "
        "--sglang-dsa-topk-backend torch --sglang-chunked-prefill-size 4096 --sglang-context-length 8192 "
        "--sglang-max-prefill-tokens 4096 --sglang-enable-fp32-moe-router "
        "--sglang-enable-deterministic-inference --sglang-disable-prefill-cuda-graph "
        "--sglang-cuda-graph-max-bs-decode 64 --sglang-watchdog-timeout 7200 --sglang-dist-timeout 1800 "
        "--sglang-trust-remote-code",
        # DSA indexer
        "--dsa --index-kd-loss-coeff 0 --index-dsa-use-layernorm --index-absorb-kv-norm-fp32 "
        "--index-use-fused-bwd-tilelang --index-use-torch-topk --index-num-attention-heads 32 "
        "--freeze-indexer --index-topk-freq 4 --index-skip-topk-offset 3 --fused-select-topk-block-size 1024",
        # misc + deterministic mode + in-process alignment assertion
        "--attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 --attention-backend flash "
        "--moe-token-dispatcher-type flex --moe-enable-deepep "
        '--train-env-vars {"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","CUDA_LAUNCH_BLOCKING":"1"} '
        "--custom-megatron-before-log-prob-hook-path "
        "slime.backends.megatron_utils.alignment.deepgemm_forward.enable_deepgemm_all_forward "
        "--custom-megatron-before-train-step-hook-path "
        "slime.backends.megatron_utils.alignment.deepgemm_forward.enable_deepgemm_all_forward_before_train_step "
        "--megatron-deepgemm-forward-layers 0 1 2 3 4 5 --megatron-deepgemm-moe-forward-layers 3 4 5 "
        "--deterministic-mode --skip-eval-before-train "
        f"--ci-test --ci-disable-kl-checker --ci-train-rollout-logprob-abs-diff-threshold {threshold}",
    ]
    if sglang_layerwise_dump is not None:
        groups.append(
            f"--sglang-debug-tensor-dump-output-folder {sglang_layerwise_dump} "
            "--sglang-debug-tensor-dump-layers 0 1 2 3 4 5"
        )
    return " ".join(groups)


def run_gate(*, layerwise_zero: bool = False, rollout_max_response_len: int = 4096) -> None:
    sglang_root = os.environ.get("SGLANG_ROOT", DEFAULT_SGLANG_ROOT)
    megatron_root = os.environ.get("MEGATRON_ROOT", DEFAULT_MEGATRON_ROOT)
    hf_model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
    native_ckpt = os.environ.get("NATIVE_CKPT", DEFAULT_NATIVE_CKPT)
    prompt_data = os.environ.get("PROMPT_DATA", DEFAULT_PROMPT_DATA)
    threshold = os.environ.get("MAX_TRAIN_ROLLOUT_DIFF", DEFAULT_MAX_TRAIN_ROLLOUT_DIFF)
    kv_cache_dtype = os.environ.get("SGLANG_KV_CACHE_DTYPE", "fp8_e4m3")
    if kv_cache_dtype not in {"bfloat16", "fp8_e4m3"}:
        raise ValueError("SGLANG_KV_CACHE_DTYPE must be bfloat16 or fp8_e4m3, " f"got {kv_cache_dtype!r}")

    reason = _skip_reason(sglang_root, megatron_root, hf_model, native_ckpt, prompt_data)
    if reason is not None:
        pytest.skip(f"6-layer GLM-5 deterministic gate skipped: {reason}")

    master_addr = "127.0.0.1"
    ifname = os.environ.get("MLP_SOCKET_IFNAME")
    if ifname and (ip := _iface_ipv4(ifname)):
        master_addr = ip
    master_port = os.environ.get("MASTER_PORT", "29500")

    # Deterministic env is sourced into the Ray head so every colocated actor
    # (SGLang engine + Megatron train actor) inherits the exact numerical stack,
    # matching how the standalone gate launches.
    env = {
        **os.environ,
        **_deterministic_env(sglang_root, megatron_root, kv_cache_dtype),
    }
    # sglang-router's Rust worker discovery does not currently honor
    # NO_PROXY reliably.  A configured HTTP proxy makes POST /workers
    # return 202 while the asynchronous health probe is sent through the
    # proxy and the worker never becomes routable.  This gate only talks to
    # colocated Ray/SGLang endpoints, so keep its whole child process tree
    # off external proxies regardless of whether it is launched as this
    # file or imported by the layerwise gate.
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(proxy_var, None)
    env["MASTER_ADDR"] = master_addr
    env["RAY_ADDRESS"] = f"{master_addr}:{master_port}"

    gate_name = "layerwise-zero" if layerwise_zero else "train/rollout"
    print(
        f"Running 6-layer GLM-5 deterministic {gate_name} gate " f"(logprob limit={float(threshold):g}) ...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="glm52_6layer_gate_") as tmp:
        ref_load = _native_checkpoint_view(native_ckpt, tmp)
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
            ref_load,
            prompt_data,
            threshold,
            rollout_dump,
            kv_cache_dtype,
            rollout_max_response_len=rollout_max_response_len,
            sglang_layerwise_dump=(sglang_layerwise_dump if layerwise_zero else None),
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
            # The aggregate bound is asserted inside the Megatron actor; a
            # breach raises there and fails the driver with a non-zero exit.
            code, _out = _run(
                [sys.executable, "-u", "train.py", *argv],
                env=env,
                cwd=str(REPO_ROOT),
                stream=True,
            )
            assert code == 0, f"train.py exited {code} (train/rollout bound {float(threshold):g} likely breached)"
            if layerwise_zero:
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
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
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
        f"6-layer GLM-5 deterministic {gate_name} gate PASSED " f"(logprob limit={float(threshold):g})",
        flush=True,
    )


def _run(cmd, env=None, cwd=None, check=True, stream=False):
    """Run a command; when stream=True, tee output to stdout and return (code, text)."""
    if not stream:
        r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"{cmd} failed ({r.returncode}): {r.stdout}\n{r.stderr}")
        return r.returncode, (r.stdout or "")
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    return proc.returncode, "".join(lines)


def test_glm52_6layer_deterministic_train_rollout_alignment():
    run_gate()


def test_glm52_alignment_gate_trains_all_main_model_parameters_without_r3():
    argv = _train_args(
        "/tmp/hf",
        "/tmp/native",
        "/tmp/prompts.jsonl",
        DEFAULT_MAX_TRAIN_ROLLOUT_DIFF,
        "/tmp/rollout/{rollout_id}.pt",
        "fp8_e4m3",
    ).split()

    assert "--moe-enable-deepep" in argv
    assert "--use-rollout-routing-replay" not in argv
    assert "--only-train-params-name-list" not in argv
    assert "--freeze-params-name-list" not in argv
    assert "--freeze-indexer" in argv
    assert argv[argv.index("--sglang-kv-cache-dtype") + 1] == "fp8_e4m3"


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    raise SystemExit(pytest.main([__file__, "-s"]))
