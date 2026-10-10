"""Real Codex / Claude Code → sunabako → MiMo grader → Qwen3.8-27B optimizer E2E.

Requires 8 GPUs (validated on H20 96GB), sunabako, skopeo/umoci and a usable sandbox node.
Without SUNABAKO_CLUSTER, creates an isolated local native-runtime node. RSS
test mode requires explicit SUNABAKO_ALLOW_TEST_MEMORY=1; it is not a hard cap.
All artifacts survive failure in SLIME_AGENT_TEST_RUN_DIR (a fresh directory).
"""

import argparse
import ast
import asyncio
import base64
import dataclasses
import fcntl
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_GPUS = 8
SAMPLES_PER_PROMPT = 4
TRAINING_STEPS = 2
CODEX_VERSION = "0.162.1"
CLAUDE_CODE_VERSION = "2.1.296"
MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
DATA_REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
TASKS = {
    "format-code-task-000003": "sha256:89d2302961adfe5b768b28b72e5c7a227e24e32d923f2a996b077c85fa3bc428",
    "format-code-task-000045": "sha256:fcb5d0910fc4725ca505d3cfb8e05b990140a83fe1c17f0152db8c742982feab",
}
SHL_TESTS = ("simple", "big", "by_zero", "non_power_of_two", "max", "by_max")


def free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def grader_runs(run_dir):
    # The MiMo shell grader returns its output directly; it does not create
    # /tmp/.eval.out. Read the provider's actual command/exit/output records.
    return [
        {**command, "image": json.loads((path.parent / "sandbox.json").read_text())["image"]}
        for path in (run_dir / "sandboxes").glob("*/commands.json")
        for command in json.loads(path.read_text())
        if "/tmp/mimo-tests.patch" in command["command"] and "git apply --verbose" in command["command"]
    ]


def checkpoint_ready(checkpoint):
    if not all((checkpoint / name).is_file() for name in ("config.json", "tokenizer.json", "tokenizer_config.json")):
        return False
    index = checkpoint / "model.safetensors.index.json"
    if not index.is_file():
        return (checkpoint / "model.safetensors").is_file()
    return all((checkpoint / name).is_file() for name in set(json.loads(index.read_text())["weight_map"].values()))


def prepare_assets():
    cache = Path(os.environ.get("SLIME_AGENT_TEST_CACHE", "/root/.cache/slime-agent-e2e"))
    cache.mkdir(parents=True, exist_ok=True)
    # Preparation now runs before GPU locking, so simultaneous CI jobs must
    # serialize writers to the model, dataset and CLI caches too.
    with (cache / "prepare.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print("[agent-assets] Waiting for another asset cache writer", flush=True)
                time.sleep(20)
        return _prepare_assets(cache)


def _prepare_assets(cache):
    from ci.oci_cache import download, pull
    from examples.coding_agent_rl.prepare_mimo import convert
    from huggingface_hub import snapshot_download

    checkpoint = Path(os.environ.get("HF_CHECKPOINT", "/root/models/Qwen3.8-27B"))
    if not checkpoint_ready(checkpoint):
        print(f"[agent-assets] Preparing model checkpoint: {checkpoint}", flush=True)
        snapshot_download(
            MODEL,
            revision=MODEL_REVISION,
            local_dir=checkpoint,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
        )
    assert checkpoint_ready(checkpoint), "Model checkpoint is incomplete"
    data = Path(os.environ.get("SLIME_AGENT_TEST_DATA", str(cache / "mimo")))
    if not all((data / name).exists() for name in ("code.parquet", "image-mapping.jsonl")):
        snapshot_download(
            DATASET,
            repo_type="dataset",
            revision=DATA_REVISION,
            local_dir=data,
            allow_patterns=["code.parquet", "image-mapping.jsonl"],
        )
    selected = {row["label"]: row for row in convert(data, list(TASKS))}
    rows = [selected[instance] for instance in TASKS]

    version = os.environ.get("SLIME_AGENT_CODEX_VERSION", CODEX_VERSION)
    archive = os.environ.get("SLIME_AGENT_CODEX_NATIVE_TARBALL")
    if not archive:
        metadata = cache / f"codex-{version}-linux-x64.metadata.json"
        if not metadata.exists():
            with urllib.request.urlopen(
                f"https://registry.npmjs.org/@openai/codex/{version}-linux-x64", timeout=60
            ) as response:
                package = json.load(response)
            temporary = metadata.with_suffix(".partial")
            temporary.write_text(json.dumps(package))
            temporary.replace(metadata)
        package = json.loads(metadata.read_text())
        archive = cache / f"codex-{version}-linux-x64.tgz"
        algorithm, digest = package["dist"]["integrity"].split("-", 1)
        download(
            lambda: urllib.request.Request(package["dist"]["tarball"]),
            archive,
            base64.b64decode(digest).hex(),
            package.get("archive_size"),
            algorithm=algorithm,
        )
        package["archive_size"] = archive.stat().st_size
        metadata.write_text(json.dumps(package))
    os.environ["SLIME_AGENT_CODEX_NATIVE_TARBALL"] = str(Path(archive).resolve())

    cc_version = os.environ.get("SLIME_AGENT_CC_VERSION", CLAUDE_CODE_VERSION)
    cc_archive = os.environ.get("SLIME_AGENT_CC_NATIVE_TARBALL")
    if not cc_archive:
        metadata = cache / f"claude-code-{cc_version}-linux-x64.metadata.json"
        if not metadata.exists():
            with urllib.request.urlopen(
                f"https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/{cc_version}", timeout=60
            ) as response:
                package = json.load(response)
            temporary = metadata.with_suffix(".partial")
            temporary.write_text(json.dumps(package))
            temporary.replace(metadata)
        package = json.loads(metadata.read_text())
        cc_archive = cache / f"claude-code-{cc_version}-linux-x64.tgz"
        algorithm, digest = package["dist"]["integrity"].split("-", 1)
        download(
            lambda: urllib.request.Request(package["dist"]["tarball"]),
            cc_archive,
            base64.b64decode(digest).hex(),
            package.get("archive_size"),
            algorithm=algorithm,
        )
        package["archive_size"] = cc_archive.stat().st_size
        metadata.write_text(json.dumps(package))
    os.environ["SLIME_AGENT_CC_NATIVE_TARBALL"] = str(Path(cc_archive).resolve())

    images = None
    if not os.environ.get("SUNABAKO_CLUSTER"):
        images = {}
        for row in rows:
            instance = row["label"]
            bundle = cache / instance
            image_name = row["metadata"]["image"]
            pull(image_name.rsplit(":", 1)[0] + "@" + TASKS[instance], bundle, cache / "oci")
            info = json.loads((bundle / "image.json").read_text())
            assert info["manifest_digest"] == TASKS[instance]
            info.update(rootfs=str(bundle / "rootfs"), workdir=row["metadata"]["workdir"])
            images[image_name] = info
    print("[agent-assets] Model, dataset, Codex / Claude Code archives and two task images ready", flush=True)
    return checkpoint, version, rows, images


def prepare(run_dir):
    from sunabako import Node

    checkpoint, version, rows, image_map = prepare_assets()
    (run_dir / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if image_map is not None:
        images = run_dir / "images.json"
        images.write_text(json.dumps(image_map))
        node = Node(state_dir=str(run_dir / "node"), rootfs=next(iter(image_map.values()))["rootfs"])
        node.configure(
            memory_capacity_bytes=32 * 1024**3,
            max_sandboxes=8,
            uid_start=100000,
            uid_range_size=65536,
            memory_overcommit=1.0,
            cgroup_parent=os.environ.get("SUNABAKO_CGROUP_PARENT"),
            allow_unbounded_memory_for_tests=os.environ.get("SUNABAKO_ALLOW_TEST_MEMORY") == "1",
            proot_binary=os.environ.get("SUNABAKO_PROOT_BINARY", "/usr/local/libexec/sunabako/proot"),
        )
        cluster = run_dir / "cluster.json"
        cluster.write_text(json.dumps({"nodes": [dataclasses.asdict(node)]}))
        os.environ.update(SUNABAKO_CLUSTER=str(cluster), SUNABAKO_IMAGES=str(images))
        os.environ.setdefault("SUNABAKO_RUNTIME", "native")
    assert os.environ.get("SUNABAKO_IMAGES"), "SUNABAKO_IMAGES is required with an existing cluster"
    os.environ.update(
        SWE_AGENT="codex",
        SWE_SANDBOX_PROVIDER="sunabako",
        SWE_TRAIN_PROTOCOL="scaleswe",
        SWE_BOOT_CONCURRENCY="8",
        SWE_AGENT_TIME_BUDGET_SEC="600",
        SWE_EVAL_TIMEOUT_SEC="180",
        SWE_ROLLOUT_GUARD_SEC="900",
        SUNABAKO_MEMORY_MB="4096",
        SLIME_AGENT_CC_EXTRA_ARGS="--tools Bash,Read,Edit,Write,Glob,Grep --disable-slash-commands",
        SLIME_AGENT_CC_EXTRA_ENVS=json.dumps(
            {"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "65536", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"}
        ),
        SUNABAKO_ARTIFACTS=str(run_dir / "sandboxes"),
        SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS="0",
        ADAPTER_PORT=str(free_port()),
        ADAPTER_BIND_HOST="0.0.0.0",
        SLIME_AGENT_TEST_RUN_DIR=str(run_dir),
        SWE_CC_PROMPT="This is a small, time-bounded repository repair. Read PROBLEM_STATEMENT.md, inspect the relevant "
        "implementation and nearby tests, make the smallest source change, then run focused tests and summarize the "
        "result before exiting promptly. Preserve existing defaults and public behavior except for the requested "
        "change; do not add features or broaden defaults. Use only the checked-out repository and existing environment: do not fetch "
        "upstream source or releases, install packages, or research unrelated history. Do not change tests, "
        "documentation, or PROBLEM_STATEMENT.md. Exclude .harness from searches; it contains live execution logs. "
        "Do not commit.",
    )
    os.environ.setdefault("ADAPTER_PUBLIC_HOST", "auto")
    # A fresh, unchanged image must fail the same hidden tests used for reward.
    from examples.coding_agent_rl import swe

    from slime.utils.types import Sample

    baselines = {}
    for row in rows:
        baseline = asyncio.run(swe.run_evaluation(swe.get_metadata(Sample(**row)), diff_text="", timeout_sec=180))
        assert baseline.reward == 0, "The fixture is already solved, or the grader is not detecting the bug"
        baselines[row["label"]] = baseline._asdict()
        records = [r for r in grader_runs(run_dir) if r["image"] == row["metadata"]["image"]]
        assert len(records) == 1 and 0 < records[0]["exit_code"] < 124
        if row["label"] == "format-code-task-000003":
            assert records[0]["exit_code"] == 2 and "cannot import name 'SHL'" in records[0]["stdout"]
        (run_dir / f"grader-baseline-{row['label']}.log").write_text(records[0]["stdout"] + records[0]["stderr"])
    (run_dir / "baseline.json").write_text(json.dumps(baselines, indent=2))
    return checkpoint, version


def execute(run_dir, checkpoint):
    import ray
    from ray.job_submission import JobStatus, JobSubmissionClient

    # Own a fresh Ray head, rather than stopping or reusing another test's job.
    host = os.environ.get("MASTER_ADDR") or ray.util.get_node_ip_address()
    proxy_bypass = ",".join(filter(None, ["localhost,127.0.0.1", host, os.environ.get("no_proxy")]))
    os.environ.update(no_proxy=proxy_bypass, NO_PROXY=proxy_bypass)
    environment = {
        "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'tests'}:{os.environ.get('MEGATRON_DIR', '/root/Megatron-LM')}",
        "PYTHONUNBUFFERED": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "RAY_USE_UVLOOP": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MASTER_ADDR": host,
        "SLIME_HOST_IP": host,
        **{
            k: v
            for k, v in os.environ.items()
            if k.startswith(("SUNABAKO_", "SWE_", "SLIME_AGENT_", "ADAPTER_"))
            or k
            in {
                "http_proxy",
                "https_proxy",
                "no_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
                "GLOO_SOCKET_IFNAME",
                "NCCL_SOCKET_IFNAME",
                "TP_SOCKET_IFNAME",
                "SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS",
                "TILELANG_CACHE_DIR",
                "TRITON_CACHE_DIR",
            }
        },
    }
    model_args = (
        subprocess.check_output(
            [
                "bash",
                "-c",
                'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"',
                "_",
                str(REPO_ROOT / "scripts/models/qwen3.5-27B.sh"),
            ]
        )
        .decode()
        .strip("\0")
        .split("\0")
    )
    # Coarser masked padding limits shape-specific training-kernel compilation
    # while retaining every original token from every agent turn.
    flags = shlex.split("""
        --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --colocate
        --custom-generate-function-path agent_e2e_helpers.generate
        --custom-megatron-before-train-step-hook-path agent_e2e_helpers.before_train_step
        --input-key prompt --label-key label --metadata-key metadata
        --num-rollout 2 --rollout-batch-size 2 --n-samples-per-prompt 4 --num-steps-per-rollout 1
        --global-batch-size 8 --micro-batch-size 1 --rollout-max-context-len 65536 --rollout-max-response-len 4096
        --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-stop-token-ids 248046 248044
        --tensor-model-parallel-size 2 --pipeline-model-parallel-size 4 --context-parallel-size 1 --sequence-parallel
        --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
        --use-dynamic-batch-size --max-tokens-per-gpu 16384 --log-probs-chunk-size 1024
        --data-pad-size-multiplier 1024
        --advantage-estimator grpo --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0
        --use-score-centering
        --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.98
        --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
        --rollout-num-gpus 4 --rollout-num-gpus-per-engine 4 --sglang-mem-fraction-static 0.45 --sglang-context-length 65536
        --sglang-max-running-requests 8 --sglang-cuda-graph-max-bs-decode 8
        --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3
        --sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 3
        --sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 4
        --sglang-mamba-scheduler-strategy extra_buffer
        --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32 --attention-backend flash
    """)
    flags += [
        "--hf-checkpoint",
        str(checkpoint),
        "--load",
        str(checkpoint),
        "--prompt-data",
        str(run_dir / "train.jsonl"),
        "--apply-chat-template-kwargs",
        '{"reasoning_effort":"medium"}',
        "--save-debug-rollout-data",
        str(run_dir / "rollout_{rollout_id}.pt"),
        "--save-debug-train-data",
        str(run_dir / "train_{rollout_id}.pt"),
    ]
    client, job = None, None
    # Match worker imports before spending time loading the 27B model. Some
    # environments contain an unrelated installed package named ``tests``.
    subprocess.run(
        [sys.executable, "-c", "from agent_e2e_helpers import generate, before_train_step"],
        env={**os.environ, **environment},
        check=True,
    )
    try:
        context = ray.init(
            address="local",
            num_gpus=NUM_GPUS,
            num_cpus=32,
            include_dashboard=True,
            dashboard_host="127.0.0.1",
            dashboard_port=free_port(),
            _node_ip_address=host,
            _temp_dir=tempfile.mkdtemp(prefix="slime-agent-ray-"),
            object_store_memory=2 * 1024**3,
        )
        client = JobSubmissionClient("http://" + context.address_info["webui_url"])
        job = client.submit_job(
            entrypoint=shlex.join([sys.executable, "-u", str(REPO_ROOT / "train.py"), *model_args, *flags]),
            runtime_env={"env_vars": environment},
        )
        deadline = time.monotonic() + int(os.environ.get("SLIME_AGENT_TEST_TIMEOUT", "1800"))
        while True:
            status = client.get_job_status(job)
            (run_dir / "train.log").write_text(client.get_job_logs(job))
            if status.is_terminal():
                assert status == JobStatus.SUCCEEDED, f"Training {status}: see {run_dir / 'train.log'}"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Agent training exceeded its deadline; see {run_dir / 'train.log'}")
            print(f"Agent training job {job}: {status}; logs: {run_dir / 'train.log'}", flush=True)
            time.sleep(min(20, remaining))
    finally:
        if client and job:
            try:
                if not client.get_job_status(job).is_terminal():
                    client.stop_job(job)
                (run_dir / "train.log").write_text(client.get_job_logs(job))
            finally:
                ray.shutdown()
        else:
            ray.shutdown()


def cleanup_sandboxes(run_dir):
    from sunabako import Cluster, Sandbox

    manifests = [json.loads(p.read_text()) for p in (run_dir / "sandboxes").glob("*/sandbox.json")]
    if not manifests:
        return
    for node in Cluster.from_file(os.environ["SUNABAKO_CLUSTER"]).nodes:
        owned = {m["sandbox_id"] for m in manifests if m["node"] == node.name}
        live = {s["spec"]["id"] for s in node.call("list")}
        for sandbox_id in owned & live:
            Sandbox.connect(sandbox_id, node=node).kill()


def verify_rollout(run_dir, rollout_id):
    import torch
    from agent_e2e_helpers import audit_grpo_training

    samples = torch.load(run_dir / f"rollout_{rollout_id}.pt", weights_only=False)["samples"]
    indices = sorted({s["index"] for s in samples})
    assert len(indices) == len(TASKS) * SAMPLES_PER_PROMPT
    assert len({s["group_index"] for s in samples}) == len(TASKS)
    assert {s["metadata"]["instance_id"] for s in samples} == set(TASKS)
    expected_agent = ("codex", "claude_code")[rollout_id]
    assert {s["metadata"]["agent"] for s in samples} == {expected_agent}
    for instance in TASKS:
        group = [s for s in samples if s["metadata"]["instance_id"] == instance]
        assert len({s["index"] for s in group}) == SAMPLES_PER_PROMPT
        assert len({s["group_index"] for s in group}) == 1, "Normalize each task's four outcomes separately"
    full_count, token_audits = 0, []
    for index in indices:
        agent_dir = run_dir / "agents" / str(index)
        full = torch.load(agent_dir / "agent-full.pt", weights_only=False)["samples"]
        selected = [s for s in samples if s["index"] == index]
        assert full and len(selected) == len(full), "Every real agent segment must enter training"
        for chosen, original in zip(selected, full, strict=True):
            for field in ("tokens", "loss_mask", "rollout_log_probs", "reward", "index", "group_index", "rollout_id"):
                assert chosen[field] == original[field], f"CI selection changed the agent's {field}"
            for field in ("rollout_top_p_token_ids", "rollout_top_p_token_offsets", "rollout_top_p_log_probs"):
                assert torch.equal(torch.as_tensor(chosen[field]), torch.as_tensor(original[field])), field
        full_count += len(full)
        token_audits.append(json.loads((agent_dir / "token-audit.json").read_text()))
    trainable_by_rollout = {}
    for sample in samples:
        assert sample["metadata"]["agent_exit_code"] == 0 and not sample["remove_sample"]
        assert sample["response_length"] == len(sample["loss_mask"]) == len(sample["rollout_log_probs"])
        assert all(math.isfinite(v) for v in sample["rollout_log_probs"])
        rid = sample["rollout_id"]
        trainable_by_rollout[rid] = trainable_by_rollout.get(rid, 0) + sum(sample["loss_mask"])
    assert all(count > 0 for count in trainable_by_rollout.values())
    trained_samples = torch.load(run_dir / f"train_{rollout_id}.pt", weights_only=False)["samples"]
    assert len(trained_samples) == len(samples)
    for trained in trained_samples:
        original = samples[trained["rollout_position"]]
        assert trained["tokens"].tolist() == original["tokens"], "Training re-tokenized the model output"
        assert trained["loss_masks"].tolist() == original["loss_mask"]
        expected_logprobs = torch.tensor(original["rollout_log_probs"], dtype=trained["rollout_log_probs"].dtype)
        assert torch.equal(trained["rollout_log_probs"].cpu(), expected_logprobs)
        for field in ("rollout_top_p_token_ids", "rollout_top_p_token_offsets", "rollout_top_p_log_probs"):
            assert torch.equal(torch.as_tensor(trained[field]).cpu(), torch.as_tensor(original[field])), field
        assert trained["rollout_ids"] == original["rollout_id"]
        assert (
            trained["rollout_mask_sums"].item() == trainable_by_rollout[original["rollout_id"]]
        ), "Forks must share their own whole-rollout denominator"
    grpo_audit = audit_grpo_training(samples, trained_samples)
    (run_dir / f"grpo-audit-{rollout_id}.json").write_text(json.dumps(grpo_audit, indent=2) + "\n")
    for audit in token_audits:
        assert audit["agent"] == expected_agent
        assert audit["exact_input_output_ids_and_logprobs"] and audit["every_sampled_token_retained_once"]
        assert audit["sampling_params"]["top_p"] == 0.95
        assert audit["sampling_params"].get("top_k", -1) == -1
        assert set(audit["replay_metadata_fields_verified"]) >= {
            "rollout_top_p_token_ids",
            "rollout_top_p_token_offsets",
            "rollout_top_p_log_probs",
        }
    return {
        "rollout_id": rollout_id,
        "agent": expected_agent,
        "instance_ids": list(TASKS),
        "sample_indices": indices,
        "agent_segments": full_count,
        "training_segments": len(samples),
        "trainable_tokens": sum(trainable_by_rollout.values()),
        "token_audits": token_audits,
        "grpo_audit": grpo_audit,
        "training_tensor_identity_verified": True,
    }


def verify(run_dir, version, timings):
    from sunabako import Cluster

    rollouts = [verify_rollout(run_dir, rollout_id) for rollout_id in range(TRAINING_STEPS)]
    updates = [json.loads(path.read_text()) for path in run_dir.glob("optimizer-rollout-*-rank-*.json")]
    expected_updates = {(rollout_id, rank) for rollout_id in range(TRAINING_STEPS) for rank in range(NUM_GPUS)}
    assert {(u["rollout_id"], u["rank"]) for u in updates} == expected_updates
    assert all(u["step_id"] == 0 and 0 <= u["grad_norm"] < math.inf for u in updates)
    for rollout in rollouts:
        if rollout["grpo_audit"]["has_learning_signal"]:
            evidence = [u for u in updates if u["rollout_id"] == rollout["rollout_id"]]
            assert all(u["grad_norm"] > 0 and u["changed_parameters"] > 0 for u in evidence)
    training_log = (run_dir / "train.log").read_text()
    acceptance = [
        (float(length), float(rate))
        for length, rate in re.findall(r"accept len: ([\d.]+), accept rate: ([\d.]+)", training_log)
    ]
    assert acceptance and any(length > 1 and rate > 0 for length, rate in acceptance), "No accepted MTP draft tokens"
    metrics = []
    for line in training_log.splitlines():
        if "'train/sc_correction':" not in line:
            continue
        match = re.search(r"step (\d+): (\{.*\})", line)
        if match:
            metrics.append({"global_step": int(match[1]), **ast.literal_eval(match[2])})
    assert sorted(m["global_step"] for m in metrics) == list(range(TRAINING_STEPS))
    for metric in metrics:
        for field in ("train/sc_correction", "train/train_rollout_logprob_abs_diff", "train/grad_norm"):
            assert math.isfinite(metric[field]), field
        assert "train/sc_centered_correction" not in metric
    manifests = [json.loads(p.read_text()) for p in (run_dir / "sandboxes").glob("*/sandbox.json")]
    trajectories = list((run_dir / "sandboxes").glob("*/trajectory.jsonl"))
    all_audits = [json.loads(p.read_text()) for p in (run_dir / "agents").glob("*/token-audit.json")]
    assert (
        len(trajectories) == len(all_audits) == TRAINING_STEPS * len(TASKS) * SAMPLES_PER_PROMPT
    ), "Each of two steps must sample four real agent runs per image, without extra resampling"
    tool_calls = {"codex": 0, "claude_code": 0}
    completed_runs = {"codex": 0, "claude_code": 0}
    for path in trajectories:
        events = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
        if any(e.get("type") == "thread.started" for e in events):
            name = "codex"
            calls = [
                e
                for e in events
                if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "command_execution"
            ]
            assert calls and any(e["type"] == "turn.completed" for e in events)
        else:
            name = "claude_code"
            calls = [
                block
                for e in events
                if e.get("type") == "assistant"
                for block in e.get("message", {}).get("content", [])
                if block.get("type") == "tool_use"
            ]
            assert calls and any(e.get("type") == "result" and not e.get("is_error", False) for e in events)
        completed_runs[name] += 1
        tool_calls[name] += len(calls)
    assert completed_runs == {"codex": 8, "claude_code": 8}
    commands = "\n".join(p.read_text() for p in (run_dir / "sandboxes").glob("*/commands.json"))
    assert f"codex-cli {version}" in commands, "Unexpected CLI version"
    cc_version = os.environ.get("SLIME_AGENT_CC_VERSION", CLAUDE_CODE_VERSION)
    assert f"{cc_version} (Claude Code)" in commands, "Unexpected Claude Code version"
    grading = grader_runs(run_dir)
    assert len(grading) == len(all_audits) + len(TASKS), "Missing independent grading for an agent run"
    passed_runs = [r for r in grading if r["exit_code"] == 0]
    assert len(passed_runs) == sum(audit["reward"] == 1 for audit in all_audits)
    for result in passed_runs:
        if result["image"].endswith(":format-code-task-000003"):
            assert all(f"test_shl_{case} PASSED" in result["stdout"] for case in SHL_TESTS)
    (run_dir / "grader-success.log").write_text("\n".join(run["stdout"] for run in passed_runs))
    owned = {m["sandbox_id"] for m in manifests}
    for node in Cluster.from_file(os.environ["SUNABAKO_CLUSTER"]).nodes:
        assert not owned.intersection(s["spec"]["id"] for s in node.call("list")), "Sandbox leaked after evaluation"
    result = {
        "model": MODEL,
        "codex_version": version,
        "claude_code_version": cc_version,
        "instance_ids": list(TASKS),
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "samples_per_step": len(TASKS) * SAMPLES_PER_PROMPT,
        "training_steps": TRAINING_STEPS,
        "inference_mtp": {
            "algorithm": "EAGLE",
            "num_steps": 3,
            "num_draft_tokens": 4,
            "observed_decode_batches": len(acceptance),
            "max_accept_length": max(length for length, _ in acceptance),
            "max_accept_rate": max(rate for _, rate in acceptance),
        },
        "rollouts": rollouts,
        "agent_tool_calls": tool_calls,
        "completed_agent_runs": completed_runs,
        "optimizer_updates": updates,
        "model_parameters_changed": any(u["changed_parameters"] > 0 for u in updates),
        "training_metrics": metrics,
        "sandbox_memory_mode": "rss-test-only" if os.environ.get("SUNABAKO_ALLOW_TEST_MEMORY") == "1" else "cgroup",
        "passed": True,
        "timings_seconds": timings,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Cache downloads without taking GPU locks")
    args = parser.parse_args()
    if args.prepare_only:
        prepare_assets()
        sys.exit(0)
    output = os.environ.get("SLIME_AGENT_TEST_RUN_DIR")
    if output:
        run_dir = Path(output).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = Path(tempfile.mkdtemp(prefix="slime-agent-e2e-"))
    os.chdir(REPO_ROOT)
    print(f"Agent E2E artifacts: {run_dir}", flush=True)
    start = time.monotonic()
    try:
        checkpoint, version = prepare(run_dir)
        prepared = time.monotonic()
        execute(run_dir, checkpoint)
        trained = time.monotonic()
        timings = {"prepare": prepared - start, "train": trained - prepared, "total": trained - start}
        (run_dir / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        verify(run_dir, version, timings)
    finally:
        cleanup_sandboxes(run_dir)
