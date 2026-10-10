"""Real Codex → sunabako → MiMo grader → Qwen3.8-27B optimizer E2E.

Requires 8 GPUs (H100 80GB), sunabako, skopeo/umoci and a usable sandbox node.
Without SUNABAKO_CLUSTER, creates an isolated local native-runtime node. RSS
test mode requires explicit SUNABAKO_ALLOW_TEST_MEMORY=1; it is not a hard cap.
All artifacts survive failure in SLIME_AGENT_TEST_RUN_DIR (a fresh directory).
"""

import ast
import asyncio
import base64
import dataclasses
import hashlib
import json
import math
import os
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
CODEX_VERSION = "0.162.1"
MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
DATA_REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
INSTANCE = "format-code-task-000003"
IMAGE_DIGEST = "sha256:89d2302961adfe5b768b28b72e5c7a227e24e32d923f2a996b077c85fa3bc428"
SHL_TESTS = ("simple", "big", "by_zero", "non_power_of_two", "max", "by_max")


def free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def grader_runs(run_dir):
    # The MiMo shell grader returns its output directly; it does not create
    # /tmp/.eval.out. Read the provider's actual command/exit/output records.
    return [
        command
        for path in (run_dir / "sandboxes").glob("*/commands.json")
        for command in json.loads(path.read_text())
        if "bash /testbed/mimo_test_command.sh" in command["command"] and "/tmp/mimo-tests.patch" in command["command"]
    ]


def prepare(run_dir):
    from examples.coding_agent_rl.prepare_mimo import convert
    from huggingface_hub import snapshot_download
    from sunabako import Node
    from sunabako.images import pull

    cache = Path(os.environ.get("SLIME_AGENT_TEST_CACHE", "/root/.cache/slime-agent-e2e"))
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(os.environ.get("HF_CHECKPOINT", "/root/models/Qwen3.8-27B"))
    if not (checkpoint / "config.json").exists():
        snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=checkpoint)
    data = Path(os.environ.get("SLIME_AGENT_TEST_DATA", str(cache / "mimo")))
    if not all((data / name).exists() for name in ("code.parquet", "image-mapping.jsonl")):
        snapshot_download(
            DATASET,
            repo_type="dataset",
            revision=DATA_REVISION,
            local_dir=data,
            allow_patterns=["code.parquet", "image-mapping.jsonl"],
        )
    row = convert(data, [INSTANCE])[0]
    (run_dir / "train.jsonl").write_text(json.dumps(row) + "\n")

    version = os.environ.get("SLIME_AGENT_CODEX_VERSION", CODEX_VERSION)
    archive = os.environ.get("SLIME_AGENT_CODEX_NATIVE_TARBALL")
    if not archive:
        with urllib.request.urlopen(
            f"https://registry.npmjs.org/@openai/codex/{version}-linux-x64", timeout=60
        ) as response:
            package = json.load(response)
        archive = cache / f"codex-{version}-linux-x64.tgz"
        if not archive.exists():
            temporary = archive.with_suffix(".partial")
            urllib.request.urlretrieve(package["dist"]["tarball"], temporary)
            temporary.replace(archive)
        algorithm, digest = package["dist"]["integrity"].split("-", 1)
        actual = base64.b64encode(hashlib.new(algorithm, archive.read_bytes()).digest()).decode()
        assert actual == digest, "Codex archive integrity check failed"
    os.environ["SLIME_AGENT_CODEX_NATIVE_TARBALL"] = str(Path(archive).resolve())

    if not os.environ.get("SUNABAKO_CLUSTER"):
        bundle = cache / INSTANCE
        if not (bundle / "image.json").exists():
            pull(row["metadata"]["image"].rsplit(":", 1)[0] + "@" + IMAGE_DIGEST, bundle)
        image = json.loads((bundle / "image.json").read_text())
        assert image["manifest_digest"] == IMAGE_DIGEST
        image.update(rootfs=str(bundle / "rootfs"), workdir=row["metadata"]["workdir"])
        images = run_dir / "images.json"
        images.write_text(json.dumps({row["metadata"]["image"]: image}))
        node = Node(state_dir=str(run_dir / "node"), rootfs=image["rootfs"])
        node.configure(
            memory_capacity_bytes=4 * 1024**3,
            max_sandboxes=2,
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
        SWE_BOOT_CONCURRENCY="1",
        SWE_AGENT_TIME_BUDGET_SEC="180",
        SWE_EVAL_TIMEOUT_SEC="180",
        SWE_ROLLOUT_GUARD_SEC="420",
        SUNABAKO_MEMORY_MB="2048",
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

    baseline = asyncio.run(swe.run_evaluation(swe.get_metadata(Sample(**row)), diff_text="", timeout_sec=180))
    assert baseline.reward == 0, "The fixture is already solved, or the grader is not detecting the bug"
    (run_dir / "baseline.json").write_text(json.dumps(baseline._asdict()))
    baseline_runs = grader_runs(run_dir)
    assert len(baseline_runs) == 1 and baseline_runs[0]["exit_code"] == 2
    assert "cannot import name 'SHL'" in baseline_runs[0]["stdout"], "Baseline failed for an unexpected reason"
    (run_dir / "grader-baseline.log").write_text(baseline_runs[0]["stdout"])
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
    flags = shlex.split(
        """
        --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --colocate
        --custom-generate-function-path agent_e2e_helpers.generate
        --custom-megatron-before-train-step-hook-path agent_e2e_helpers.before_train_step
        --input-key prompt --label-key label --metadata-key metadata
        --num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1 --num-steps-per-rollout 1
        --global-batch-size 1 --micro-batch-size 1 --rollout-max-context-len 32768 --rollout-max-response-len 4096
        --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-stop-token-ids 248046 248044
        --tensor-model-parallel-size 2 --pipeline-model-parallel-size 4 --context-parallel-size 1 --sequence-parallel
        --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
        --use-dynamic-batch-size --max-tokens-per-gpu 16384 --log-probs-chunk-size 1024
        --advantage-estimator grpo --disable-rewards-normalization --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0
        --use-score-centering
        --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.98
        --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
        --rollout-num-gpus 4 --rollout-num-gpus-per-engine 4 --sglang-mem-fraction-static 0.45 --sglang-context-length 32768
        --sglang-max-running-requests 2 --sglang-cuda-graph-max-bs-decode 4
        --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3
        --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32 --attention-backend flash
    """
    )
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
        deadline = time.monotonic() + int(os.environ.get("SLIME_AGENT_TEST_TIMEOUT", "600"))
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


def verify(run_dir, version, timings):
    import torch
    from sunabako import Cluster

    samples = torch.load(run_dir / "rollout_0.pt", weights_only=False)["samples"]
    full_samples = torch.load(run_dir / "agent-full.pt", weights_only=False)["samples"]
    assert len(samples) == 2 and len(full_samples) >= 2
    for selected, original in zip(samples, [full_samples[0], full_samples[-1]], strict=True):
        for field in ("tokens", "loss_mask", "rollout_log_probs", "reward", "index", "group_index", "rollout_id"):
            assert selected[field] == original[field], f"CI selection changed the agent's {field}"
        for field in ("rollout_top_p_token_ids", "rollout_top_p_token_offsets", "rollout_top_p_log_probs"):
            assert torch.equal(torch.as_tensor(selected[field]), torch.as_tensor(original[field])), field
    assert samples and {s["index"] for s in samples} == {0}
    assert any(s["reward"] == 1 for s in samples), "No independently graded reward=1 rollout"
    trainable = 0
    for sample in samples:
        assert sample["metadata"]["agent_exit_code"] == 0 and not sample["remove_sample"]
        assert sample["response_length"] == len(sample["loss_mask"]) == len(sample["rollout_log_probs"])
        assert all(math.isfinite(v) for v in sample["rollout_log_probs"])
        trainable += sum(sample["loss_mask"])
    assert trainable > 0 and (run_dir / "train_0.pt").exists()
    trained_samples = torch.load(run_dir / "train_0.pt", weights_only=False)["samples"]
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
        assert trained["rollout_mask_sums"].item() == trainable, "Forks must share the whole-rollout denominator"
    token_audit = json.loads((run_dir / "token-audit.json").read_text())
    assert token_audit["exact_input_output_ids_and_logprobs"] and token_audit["every_sampled_token_retained_once"]
    assert token_audit["sampling_params"]["top_p"] == 0.95
    assert token_audit["sampling_params"].get("top_k", -1) == -1
    assert set(token_audit["replay_metadata_fields_verified"]) >= {
        "rollout_top_p_token_ids",
        "rollout_top_p_token_offsets",
        "rollout_top_p_log_probs",
    }
    updates = [json.loads(path.read_text()) for path in run_dir.glob("optimizer-rank-*.json")]
    assert {u["rank"] for u in updates} == set(range(NUM_GPUS)), "Missing optimizer evidence on a training rank"
    assert all(u["changed_parameters"] > 0 and 0 < u["grad_norm"] < math.inf for u in updates)
    assert all(u["rollout_id"] == 0 and u["step_id"] == 0 for u in updates), "Expected exactly one optimizer step"
    metrics = [
        ast.literal_eval(line.split("step 0: ", 1)[1])
        for line in (run_dir / "train.log").read_text().splitlines()
        if "step 0: {" in line and "'train/sc_correction':" in line
    ]
    assert len(metrics) == 1, "Missing score-centering training metrics"
    metrics = metrics[0]
    for field in ("train/sc_correction", "train/train_rollout_logprob_abs_diff", "train/grad_norm"):
        assert math.isfinite(metrics[field]), field
    assert "train/sc_centered_correction" not in metrics
    manifests = [json.loads(p.read_text()) for p in (run_dir / "sandboxes").glob("*/sandbox.json")]
    trajectories = list((run_dir / "sandboxes").glob("*/trajectory.jsonl"))
    assert len(trajectories) == 1, "Expected one real Codex run"
    tool_calls = 0
    for path in trajectories:
        events = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
        calls = [
            e
            for e in events
            if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "command_execution"
        ]
        assert calls and any(e["type"] == "turn.completed" for e in events)
        tool_calls += len(calls)
    commands = "\n".join(p.read_text() for p in (run_dir / "sandboxes").glob("*/commands.json"))
    assert f"codex-cli {version}" in commands, "Unexpected CLI version"
    passed_runs = [
        r
        for r in grader_runs(run_dir)
        if r["exit_code"] == 0 and all(f"test_shl_{case} PASSED" in r["stdout"] for case in SHL_TESTS)
    ]
    assert len(passed_runs) == 1, "Official hidden tests did not pass"
    (run_dir / "grader-success.log").write_text(passed_runs[0]["stdout"])
    # Check only this run's sandbox IDs, leaving unrelated node workloads alone.
    owned = {m["sandbox_id"] for m in manifests}
    for node in Cluster.from_file(os.environ["SUNABAKO_CLUSTER"]).nodes:
        assert not owned.intersection(s["spec"]["id"] for s in node.call("list")), "Sandbox leaked after evaluation"
    result = {
        "model": MODEL,
        "codex_version": version,
        "instance_id": INSTANCE,
        "agent_segments": len(full_samples),
        "training_segments": len(samples),
        "trainable_tokens": trainable,
        "token_audit": token_audit,
        "training_tensor_identity_verified": True,
        "rewards": [s["reward"] for s in samples],
        "codex_tool_calls": tool_calls,
        "optimizer_updates": updates,
        "training_metrics": metrics,
        "sandbox_memory_mode": "rss-test-only" if os.environ.get("SUNABAKO_ALLOW_TEST_MEMORY") == "1" else "cgroup",
        "passed": True,
        "timings_seconds": timings,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
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
