"""End-to-end GRPO check of Nemotron 3.5 Super VL on a proxy model, on one node of 8 GPUs.

The real model (124B, 88 layers, 512 experts) needs 4 nodes, so this runs a proxy with
the real model's structure and sizes except the depth and the expert count (see
``nemotron_proxy_checkpoint.py``): a Mamba2 / MoE / attention hybrid with the full RADIO
vision tower, random weights, built on the fly from the real checkpoint's config and
remote-code files. It drives the same path as the real recipe -- HF checkpoint loaded
into Megatron, a colocated SGLang engine, Megatron -> SGLang weight sync, GRPO on geo3k
images -- and asserts, in process via ``--ci-test``, that training reproduces the
rollout log-probs to within ``MAX_TRAIN_ROLLOUT_DIFF``.

Layout, as in the real recipe but on 8 GPUs: TP2 x CP2 (so DP2), EP8 over all ranks, one
SGLang engine on the node (TP8 / EP8).

The weights are random, so this checks that the two sides agree and that every piece
runs, not that the model is accurate. The bound is slime's default CI bound; it has not
been measured on this proxy and should be tightened once a run gives a baseline.

It skips on a runner without 8 GPUs, without ``mamba-ssm`` / ``causal-conv1d`` (the slime
image installs them), or without a Megatron with the hybrid model.

Environment overrides (all optional):

* ``MAX_TRAIN_ROLLOUT_DIFF`` -- the bound (default ``0.1``).
* ``NEMOTRON_VL_FILES``  -- directory with the real checkpoint's non-weight files
  (``config.json``, the remote-code ``*.py``, tokenizer and processor files). Default:
  downloaded from ``nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16``.
* ``PROXY_CHECKPOINT``   -- where the proxy checkpoint is written and cached.
* ``PROMPT_DATA``        -- geo3k parquet (default: downloaded from ``chenhegu/geo3k_imgurl``).
* ``NEMOTRON_REAL_CHECKPOINT`` -- a full real checkpoint; enables one extra CPU check of
  the proxy's tensor names and shapes against its safetensors headers.
* ``MEGATRON_ROOT`` (default ``/root/Megatron-LM``), ``MLP_SOCKET_IFNAME``, ``MASTER_PORT``,
  ``RAY_ADDRESS`` -- as in ``test_glm52_6layer_deterministic_e2e.py``.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Single-node gate; run-ci-changed reads this top-level constant.
NUM_GPUS = 8

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nemotron_proxy_checkpoint import (  # noqa: E402
    ProxyShape,
    numel,
    proxy_config,
    tensor_specs,
    write_proxy_checkpoint,
)

DEFAULT_REAL_REPO = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
DEFAULT_PROXY_CHECKPOINT = "/root/models/Nemotron-3.5-Super-VL-proxy"
DEFAULT_PROMPT_REPO = "chenhegu/geo3k_imgurl"
DEFAULT_PROMPT_DATA = "/root/datasets/geo3k_imgurl/train.parquet"
DEFAULT_MEGATRON_ROOT = "/root/Megatron-LM"
MODEL_SCRIPT = REPO_ROOT / "scripts/models/nemotron3.5-super-120B-A12B.sh"

# slime's default CI bound for train_rollout_logprob_abs_diff. Not measured on this proxy.
DEFAULT_MAX_TRAIN_ROLLOUT_DIFF = "0.1"

PROXY = ProxyShape()

# Layout on one node of 8 GPUs. Dense layers see TP x CP x DP = 8; the MoE layers see
# ETP x EP x EDP = 8 with EP over every rank, as in the 4-node recipe (EP32 over 32).
TP, CP, PP, EP, ETP = 2, 2, 1, 8, 1
DP = NUM_GPUS // (TP * CP * PP)

NUM_ROLLOUT = 2
ROLLOUT_BATCH_SIZE = 4
N_SAMPLES_PER_PROMPT = 2
GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT  # one training step per rollout
MAX_RESPONSE_LEN = 256


def _model_args(shape: ProxyShape = PROXY) -> list[str]:
    """The real model's arguments from its config script, with the proxy's depth and experts last."""
    out = subprocess.run(
        ["bash", "-c", f'source "{MODEL_SCRIPT}" && printf "%s\\0" "${{MODEL_ARGS[@]}}"'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    args = [arg for arg in out.split("\0") if arg]
    # argparse keeps the last occurrence, so these override the 88-layer / 512-expert values.
    return args + [
        "--hybrid-override-pattern",
        shape.pattern,
        "--num-layers",
        str(len(shape.pattern)),
        "--num-experts",
        str(shape.num_experts),
        "--moe-router-topk",
        str(shape.topk),
    ]


def _train_args(proxy_checkpoint: str, prompt_data: str, threshold: str, rollout_dump: str) -> list[str]:
    return [
        # placement: one node, colocated train + rollout
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(NUM_GPUS),
        "--rollout-num-gpus",
        str(NUM_GPUS),
        "--colocate",
        "--train-backend",
        "megatron",
        "--multimodal-keys",
        '{"image": "images"}',
        *_model_args(),
        "--hf-checkpoint",
        proxy_checkpoint,
        "--load",
        proxy_checkpoint,
        # rollout: geo3k images, a few short samples
        "--prompt-data",
        prompt_data,
        "--input-key",
        "problem",
        "--label-key",
        "answer",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--rm-type",
        "deepscaler",
        "--num-rollout",
        str(NUM_ROLLOUT),
        "--rollout-batch-size",
        str(ROLLOUT_BATCH_SIZE),
        "--n-samples-per-prompt",
        str(N_SAMPLES_PER_PROMPT),
        "--rollout-max-response-len",
        str(MAX_RESPONSE_LEN),
        "--rollout-temperature",
        "0.8",
        "--global-batch-size",
        str(GLOBAL_BATCH_SIZE),
        "--save-debug-rollout-data",
        rollout_dump,
        # GRPO, as in the real recipe
        "--advantage-estimator",
        "grpo",
        "--use-score-centering",
        "--score-centering-top-k",
        "128",
        "--pg-loss-type",
        "reinforce",
        "--use-tis",
        "--tis-clip-low",
        "0",
        "--tis-clip",
        "2",
        "--disable-grpo-std-normalization",
        "--calculate-per-token-loss",
        "--kl-loss-coef",
        "0.00",
        "--kl-loss-type",
        "low_var_kl",
        "--kl-coef",
        "0.00",
        "--entropy-coef",
        "0.00",
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        # SGLang: one engine on the node, TP8 / EP8
        "--rollout-num-gpus-per-engine",
        str(NUM_GPUS),
        "--sglang-mem-fraction-static",
        "0.6",
        "--sglang-ep-size",
        str(NUM_GPUS),
        "--sglang-cuda-graph-bs",
        "1",
        "2",
        "4",
        "8",
        "--sglang-max-running-requests",
        "64",
        "--sglang-context-length",
        "4096",
        # Megatron: TP2 x CP2 (DP2), EP8
        "--tensor-model-parallel-size",
        str(TP),
        "--sequence-parallel",
        "--pipeline-model-parallel-size",
        str(PP),
        "--context-parallel-size",
        str(CP),
        "--expert-model-parallel-size",
        str(EP),
        "--expert-tensor-parallel-size",
        str(ETP),
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "flash",
        "--micro-batch-size",
        "1",
        # the bound, asserted in process
        "--skip-eval-before-train",
        "--ci-test",
        "--ci-disable-kl-checker",
        "--ci-train-rollout-logprob-abs-diff-threshold",
        threshold,
    ]


# ------------------------------------------------------------------ the GPU run


def _pythonpath(megatron_root: str) -> str:
    parts = [str(REPO_ROOT), megatron_root]
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(parts)


def _iface_ipv4(ifname: str) -> str | None:
    out = subprocess.run(["ip", "-o", "-4", "addr", "show", ifname], capture_output=True, text=True).stdout.split()
    for i, token in enumerate(out):
        if token == "inet" and i + 1 < len(out):
            return out[i + 1].split("/")[0]
    return None


def _skip_reason(megatron_root: str) -> str | None:
    if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode != 0:
        return "nvidia-smi not found (no GPUs)"
    names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.splitlines()
    if len([line for line in names if line.strip()]) < NUM_GPUS:
        return f"needs {NUM_GPUS} GPUs, found {len(names)}"
    probe = subprocess.run(
        [sys.executable, "-c", "import mamba_ssm, causal_conv1d; import megatron.core.models.hybrid.hybrid_model"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _pythonpath(megatron_root)},
    )
    if probe.returncode != 0:
        return f"mamba-ssm / causal-conv1d / Megatron hybrid model unavailable: {probe.stderr.strip()[-200:]}"
    return None


def _hf_download(
    repo: str, destination: str, expected_file: str, *, repo_type: str | None = None, ignore=None
) -> None:
    # The Python API, not the `hf` CLI: the CLI's `--exclude` takes one pattern per flag in newer
    # releases, so `--exclude A B C` reads B and C as file names and fetches only those (CI then
    # got the pngs and mds but no config.json). ignore_patterns means the same in every version.
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, repo_type=repo_type, local_dir=destination, ignore_patterns=ignore)
    if not Path(expected_file).exists():
        raise FileNotFoundError(f"Downloaded {repo} but {expected_file} is still missing")


def _prepare_assets(proxy_checkpoint: str, prompt_data: str) -> None:
    if not Path(prompt_data).exists():
        if prompt_data != DEFAULT_PROMPT_DATA:
            raise FileNotFoundError(f"prompt data missing: {prompt_data}")
        _hf_download(DEFAULT_PROMPT_REPO, str(Path(prompt_data).parent), prompt_data, repo_type="dataset")
    with tempfile.TemporaryDirectory(prefix="nemotron_vl_files_") as downloaded:
        files = os.environ.get("NEMOTRON_VL_FILES")
        if not files:
            _hf_download(
                DEFAULT_REAL_REPO,
                downloaded,
                str(Path(downloaded, "config.json")),
                ignore=["*.safetensors", "*.png", "*.md"],
            )
            files = downloaded
        write_proxy_checkpoint(files, proxy_checkpoint, PROXY)


def _metric_values(output: str) -> list[float]:
    return [float(v) for v in re.findall(r"'train/train_rollout_logprob_abs_diff':\s*([^,}\s]+)", output)]


def _assert_result(output: str, rollout_path: str, threshold: float, image_token_id: int) -> None:
    import torch

    values = _metric_values(output)
    assert (
        len(values) == NUM_ROLLOUT
    ), f"expected {NUM_ROLLOUT} training steps, the log reports {len(values)}: {values}"
    assert all(math.isfinite(v) and 0 <= v <= threshold for v in values), values
    samples = torch.load(rollout_path, map_location="cpu", weights_only=False)["samples"]
    assert len(samples) == GLOBAL_BATCH_SIZE, f"expected {GLOBAL_BATCH_SIZE} rollout samples, got {len(samples)}"
    for sample in samples:
        length = sample["response_length"]
        assert length > 0, "no generated tokens"
        log_probs = sample["rollout_log_probs"]
        assert len(log_probs) == length and all(math.isfinite(v) for v in log_probs)
        mask = sample.get("loss_mask")
        assert mask is None or (len(mask) == length and sum(mask) > 0), "response is entirely masked"
        assert sample["weight_versions"] and set(map(str, sample["weight_versions"])) == {
            "1"
        }, f"rollout did not use the initial synchronized weights: {sample['weight_versions']}"
    if all("tokens" in sample for sample in samples):
        assert any(image_token_id in list(sample["tokens"]) for sample in samples), "no sample carries image tokens"


def _run(cmd, env=None, cwd=None, check=True, stream=False):
    if not stream:
        r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"{cmd} failed ({r.returncode}): {r.stdout}\n{r.stderr}")
        return r.returncode, r.stdout or ""
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(f"{cmd} failed ({proc.returncode})")
    return proc.returncode, "".join(lines)


def run_proxy_e2e() -> None:
    megatron_root = os.environ.get("MEGATRON_ROOT", DEFAULT_MEGATRON_ROOT)
    proxy_checkpoint = os.environ.get("PROXY_CHECKPOINT", DEFAULT_PROXY_CHECKPOINT)
    prompt_data = os.environ.get("PROMPT_DATA", DEFAULT_PROMPT_DATA)
    threshold = os.environ.get("MAX_TRAIN_ROLLOUT_DIFF", DEFAULT_MAX_TRAIN_ROLLOUT_DIFF)

    reason = _skip_reason(megatron_root)
    if reason is not None:
        message = f"Nemotron 3.5 Super VL proxy e2e skipped: {reason}"
        print(message, flush=True)
        pytest.skip(message)
    _prepare_assets(proxy_checkpoint, prompt_data)
    image_token_id = json.loads(Path(proxy_checkpoint, "config.json").read_text())["img_context_token_id"]

    master_addr = "127.0.0.1"
    ifname = os.environ.get("MLP_SOCKET_IFNAME")
    if ifname and (address := _iface_ipv4(ifname)):
        master_addr = address
    master_port = os.environ.get("MASTER_PORT", "29500")

    nvlink = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout.count("NV")
    train_env = {
        "PYTHONPATH": _pythonpath(megatron_root),
        "PYTHONUNBUFFERED": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1" if nvlink else "0",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    if ifname:
        train_env.update({"GLOO_SOCKET_IFNAME": ifname, "NCCL_SOCKET_IFNAME": ifname})
    env = {**os.environ, **train_env, "MASTER_ADDR": master_addr}
    # The router's worker discovery does not honour NO_PROXY reliably; everything here is local.
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(proxy_var, None)
        train_env[proxy_var] = ""
    existing_ray_address = os.environ.get("RAY_ADDRESS")
    env["RAY_ADDRESS"] = existing_ray_address or f"{master_addr}:{master_port}"
    env["SLIME_PROXY_ENV"] = json.dumps(train_env)

    print(f"Running the Nemotron 3.5 Super VL proxy e2e (logprob limit={float(threshold):g}) ...", flush=True)
    with tempfile.TemporaryDirectory(prefix="nemotron_vl_proxy_") as tmp:
        rollout_dump = os.path.join(tmp, "rollout_data", "{rollout_id}.pt")
        argv = _train_args(proxy_checkpoint, prompt_data, threshold, rollout_dump)
        try:
            if not existing_ray_address:
                _run(["ray", "stop", "--force"], check=False, env=env)
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
            code, output = _run(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import json, os, ray, runpy, sys; "
                    "ray.init(address=os.environ['RAY_ADDRESS'], "
                    "runtime_env={'env_vars': json.loads(os.environ['SLIME_PROXY_ENV'])}); "
                    "sys.argv[0] = 'train.py'; runpy.run_path('train.py', run_name='__main__')",
                    *argv,
                ],
                env=env,
                cwd=str(REPO_ROOT),
                check=False,
                stream=True,
            )
            assert code == 0, f"train.py exited {code} (the {float(threshold):g} bound was likely breached)"
            _assert_result(output, rollout_dump.format(rollout_id=0), float(threshold), image_token_id)
        finally:
            if not existing_ray_address:
                _run(["ray", "stop", "--force"], check=False, env=env)
    print(f"Nemotron 3.5 Super VL proxy e2e PASSED (logprob limit={float(threshold):g})", flush=True)


def test_nemotron35_super_vl_proxy_grpo_train_rollout_alignment():
    run_proxy_e2e()


# ------------------------------------------------------- CPU checks, no GPU needed

# The real checkpoint's dimensions (config.json); tensor_specs() must reproduce its totals.
_REAL_PATTERN = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
REAL_TENSOR_COUNT = 43079
REAL_PARAMETER_COUNT = 124453395976
REAL_CONFIG = {
    "vit_hidden_size": 1280,
    "projector_hidden_size": 20480,
    "downsample_ratio": 0.5,
    "norm_mean": [0.48145466, 0.4578275, 0.40821073],
    "norm_std": [0.26862954, 0.26130258, 0.27577711],
    "img_context_token_id": 18,
    "llm_config": {
        "layers_block_type": [{"M": "mamba", "E": "moe", "*": "attention"}[s] for s in _REAL_PATTERN],
        "mtp_layers_block_type": ["attention", "moe"],
        "num_nextn_predict_layers": 1,
        "hidden_size": 4096,
        "vocab_size": 131072,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "mamba_num_heads": 128,
        "mamba_head_dim": 64,
        "n_groups": 8,
        "ssm_state_size": 128,
        "conv_kernel": 4,
        "n_routed_experts": 512,
        "num_experts_per_tok": 22,
        "moe_latent_size": 1024,
        "moe_intermediate_size": 2688,
        "moe_shared_expert_intermediate_size": 5376,
    },
    "vision_config": {
        "hidden_size": 1280,
        "num_hidden_layers": 32,
        "mlp_ratio": 4.0,
        "patch_size": 16,
        "num_channels": 3,
        "num_cls_tokens": 3,
        "num_registers": 7,
        "max_img_size": 2048,
        "video_temporal_patch_size": 2,
        "summary_idxs": [0, 1],
    },
}


@pytest.mark.unit
def test_tensor_specs_reproduce_the_real_checkpoint_totals():
    specs = tensor_specs(REAL_CONFIG)
    assert len(specs) == REAL_TENSOR_COUNT
    assert sum(numel(spec) for spec in specs.values()) == REAL_PARAMETER_COUNT


@pytest.mark.unit
def test_tensor_specs_match_the_real_checkpoint_headers():
    real = os.environ.get("NEMOTRON_REAL_CHECKPOINT")
    if not real:
        pytest.skip("set NEMOTRON_REAL_CHECKPOINT to a full real checkpoint")
    import struct

    index = json.loads(Path(real, "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads(Path(real, "config.json").read_text())
    headers = {}
    for shard in sorted(set(index.values())):
        with open(Path(real, shard), "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        headers.update({k: (tuple(v["shape"]), v["dtype"]) for k, v in header.items() if k != "__metadata__"})
    specs = {name: (tuple(spec.shape), spec.dtype) for name, spec in tensor_specs(config).items()}
    assert specs == headers


@pytest.mark.unit
def test_proxy_keeps_every_layer_type_and_is_a_prefix_of_the_real_pattern():
    assert set(PROXY.pattern) == {"M", "E", "*"}
    assert _REAL_PATTERN.startswith(PROXY.pattern)
    config = proxy_config(REAL_CONFIG, PROXY)
    assert config["llm_config"]["layers_block_type"] == PROXY.layer_types()
    assert config["llm_config"]["n_routed_experts"] == PROXY.num_experts
    assert REAL_CONFIG["llm_config"]["n_routed_experts"] == 512, "proxy_config must not edit its input"


@pytest.mark.unit
def test_proxy_is_small_and_keeps_the_vision_tower():
    specs = tensor_specs(proxy_config(REAL_CONFIG, PROXY))
    assert sum(numel(spec) for spec in specs.values()) < 4_000_000_000
    assert sum(name.startswith("vision_model.radio_model.model.blocks.") for name in specs) == 32 * 12


@pytest.mark.unit
def test_layout_fits_one_node_of_8_gpus():
    assert NUM_GPUS == 8
    assert TP * CP * PP * DP == NUM_GPUS
    assert ETP * EP <= NUM_GPUS and NUM_GPUS % (ETP * EP) == 0
    assert PROXY.num_experts % EP == 0 and PROXY.topk <= PROXY.num_experts
    assert GLOBAL_BATCH_SIZE % DP == 0
    assert GLOBAL_BATCH_SIZE == ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT


@pytest.mark.unit
def test_train_args_use_one_node_of_8_gpus_and_the_proxy_model():
    argv = _train_args("/tmp/proxy", "/tmp/prompts.parquet", DEFAULT_MAX_TRAIN_ROLLOUT_DIFF, "/tmp/r/{rollout_id}.pt")

    def value(flag: str) -> str:
        """The value argparse keeps: the one after the last occurrence."""
        return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]

    assert value("--actor-num-nodes") == "1"
    assert value("--actor-num-gpus-per-node") == str(NUM_GPUS)
    assert value("--rollout-num-gpus") == str(NUM_GPUS)
    assert value("--rollout-num-gpus-per-engine") == str(NUM_GPUS)
    assert "--colocate" in argv and "--ci-test" in argv
    assert value("--hf-checkpoint") == value("--load") == "/tmp/proxy"
    # The proxy's depth and expert count come after the real model's, so they win.
    assert value("--hybrid-override-pattern") == PROXY.pattern
    assert value("--num-layers") == str(len(PROXY.pattern))
    assert value("--num-experts") == str(PROXY.num_experts)
    assert value("--moe-router-topk") == str(PROXY.topk)
    assert argv[argv.index("--num-layers") + 1] == "88", "the real model's value should come first"
    # The VL provider from the model script, not the text one.
    assert "slime_plugins.models.nemotron_h_vl" in argv
    assert value("--multimodal-keys") == '{"image": "images"}'
    assert value("--tensor-model-parallel-size") == str(TP) and value("--context-parallel-size") == str(CP)
    assert value("--expert-model-parallel-size") == str(EP)


@pytest.mark.unit
def test_written_proxy_checkpoint_has_exactly_the_specified_tensors(tmp_path):
    pytest.importorskip("safetensors")
    pytest.importorskip("torch")
    from safetensors import safe_open

    # A miniature of the same structure, so this runs on a CPU runner in a moment.
    mini = json.loads(json.dumps(REAL_CONFIG))
    llm, vision = mini["llm_config"], mini["vision_config"]
    llm.update(
        hidden_size=32,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        mamba_num_heads=4,
        mamba_head_dim=4,
        n_groups=2,
        ssm_state_size=4,
        moe_latent_size=16,
        moe_intermediate_size=24,
        moe_shared_expert_intermediate_size=40,
    )
    vision.update(hidden_size=16, num_hidden_layers=2, max_img_size=64, num_cls_tokens=1, num_registers=1)
    mini.update(vit_hidden_size=16, projector_hidden_size=48)
    files = tmp_path / "files"
    files.mkdir()
    (files / "config.json").write_text(json.dumps(mini))
    (files / "tokenizer.json").write_text("{}")
    shape = ProxyShape(pattern="ME*", num_experts=4, topk=2)

    out = write_proxy_checkpoint(files, tmp_path / "proxy", shape, shard_bytes=64 * 1024)

    config = json.loads((out / "config.json").read_text())
    assert config["llm_config"]["layers_block_type"] == ["mamba", "moe", "attention"]
    assert (out / "tokenizer.json").exists()
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    specs = tensor_specs(config)
    assert set(index) == set(specs)
    assert len(set(index.values())) > 1, "the miniature should span several shards"
    seen = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(out / shard), framework="pt") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                seen[name] = (
                    tuple(tensor.shape),
                    {"torch.bfloat16": "BF16", "torch.float32": "F32", "torch.int64": "I64"}[str(tensor.dtype)],
                )
    assert seen == {name: (tuple(spec.shape), spec.dtype) for name, spec in specs.items()}
    # Norm scales start at 1, so the random network is numerically sane.
    with safe_open(str(out / index["language_model.backbone.norm_f.weight"]), framework="pt") as handle:
        assert handle.get_tensor("language_model.backbone.norm_f.weight").eq(1).all()
    # A second call with the same shape and seed reuses the checkpoint.
    stamp = (out / "proxy_stamp.txt").stat().st_mtime_ns
    assert write_proxy_checkpoint(files, out, shape, shard_bytes=64 * 1024) == out
    assert (out / "proxy_stamp.txt").stat().st_mtime_ns == stamp


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-rs"]))
