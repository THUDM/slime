"""E2E test for --rollout-external-engine-addrs with a mixed external fleet.

Spawns four SGLang servers out-of-band on a single 8-GPU box (all tp=1):
- 1 prefill  (``--disaggregation-mode prefill``)
- 1 decode   (``--disaggregation-mode decode``)
- 2 regular  (no disaggregation)

and points slime at all four via ``--rollout-external-engine-addrs ...``.
The remaining 4 GPUs train. slime queries ``/server_info`` on each engine to
infer per-engine TP / GPU counts and registers them to its (PD-enabled) router.
"""

import os
import subprocess
import time
import urllib.request

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3.5-0.8B"
MODEL_TYPE = "qwen3.5-0.8B"
NUM_GPUS = 8
NUM_TRAIN_GPUS = 4
NUM_REGULAR_ENGINES = 2

EXTERNAL_HOST = "127.0.0.1"
PREFILL_PORT = 13150
DECODE_PORT = 13151
REGULAR_PORTS = [13153, 13154]
BOOTSTRAP_PORT = 13152


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_TRAIN_GPUS,
        dir_dst="/dev/shm",
    )


def _get_gpu_split():
    """Partition the 8 visible GPUs: 4 train + 1 prefill + 1 decode + 2 regular."""
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(NUM_GPUS))).split(",")
    assert len(all_gpus) >= NUM_GPUS, f"Expected at least {NUM_GPUS} GPUs, got {len(all_gpus)}"
    train_gpus = all_gpus[:NUM_TRAIN_GPUS]
    cursor = NUM_TRAIN_GPUS
    prefill_gpu = all_gpus[cursor]
    cursor += 1
    decode_gpu = all_gpus[cursor]
    cursor += 1
    regular_gpus = all_gpus[cursor : cursor + NUM_REGULAR_ENGINES]
    return train_gpus, prefill_gpu, decode_gpu, regular_gpus


def _launch_sglang_server(
    *,
    gpus: list[str],
    port: int,
    tp: int,
    log_path: str,
    disaggregation_mode: str | None = None,
    disaggregation_bootstrap_port: int | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        f"/root/models/{MODEL_NAME}",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp",
        str(tp),
        "--mem-fraction-static",
        "0.6",
        "--trust-remote-code",
    ]
    if disaggregation_mode is not None:
        cmd += [
            "--disaggregation-mode",
            disaggregation_mode,
            "--disaggregation-transfer-backend",
            "mooncake",
        ]
    if disaggregation_bootstrap_port is not None:
        cmd += ["--disaggregation-bootstrap-port", str(disaggregation_bootstrap_port)]

    log_file = open(log_path, "w")
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    label = disaggregation_mode or "regular"
    print(
        f"Starting external sglang {label} server on GPUs {gpus} "
        f"port={port} tp={tp} (pid={process.pid}), log: {log_path}"
    )

    # Wait up to ~10 minutes for /server_info to come up.  /health_generate
    # is unreliable for prefill/decode-only nodes, so we poll /server_info
    # — that's what slime's discover_external_engines uses anyway.
    deadline = time.time() + 600
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} server exited with code {process.returncode}; check {log_path}")
        try:
            req = urllib.request.urlopen(f"http://{EXTERNAL_HOST}:{port}/server_info", timeout=2)
            if req.status == 200:
                print(f"External sglang {label} server is ready on GPUs {gpus}")
                return process
        except Exception:
            pass
        time.sleep(5)

    process.kill()
    raise RuntimeError(f"{label} server failed to start within timeout; check {log_path}")


def execute():
    train_gpus, prefill_gpu, decode_gpu, regular_gpus = _get_gpu_split()
    processes: list[subprocess.Popen] = []

    # Restrict CUDA_VISIBLE_DEVICES to training GPUs before Ray starts so
    # ray's bundle allocator doesn't try to claim the external sglang GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)

    def launch_external_engines():
        processes.append(
            _launch_sglang_server(
                gpus=[prefill_gpu],
                port=PREFILL_PORT,
                tp=1,
                disaggregation_mode="prefill",
                disaggregation_bootstrap_port=BOOTSTRAP_PORT,
                log_path="/tmp/sglang_external_prefill.log",
            )
        )
        processes.append(
            _launch_sglang_server(
                gpus=[decode_gpu],
                port=DECODE_PORT,
                tp=1,
                disaggregation_mode="decode",
                log_path="/tmp/sglang_external_decode.log",
            )
        )
        for idx, (gpu, port) in enumerate(zip(regular_gpus, REGULAR_PORTS, strict=True)):
            processes.append(
                _launch_sglang_server(
                    gpus=[gpu],
                    port=port,
                    tp=1,
                    log_path=f"/tmp/sglang_external_regular_{idx}.log",
                )
            )

    try:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /dev/shm/{MODEL_NAME}_torch_dist "

        rollout_args = (
            "--prompt-data /root/datasets/gsm8k/train.parquet "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            "--num-rollout 2 "
            "--rollout-batch-size 4 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 512 "
            "--rollout-temperature 0.8 "
            "--global-batch-size 16 "
        )

        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 9216 "
        )

        grpo_args = (
            "--advantage-estimator grpo "
            "--use-kl-loss "
            "--kl-loss-coef 0.00 "
            "--kl-loss-type low_var_kl "
            "--entropy-coef 0.00 "
            "--eps-clip 0.2 "
            "--eps-clip-high 0.28 "
        )

        optimizer_args = (
            "--optimizer adam "
            "--lr 1e-6 "
            "--lr-decay-style constant "
            "--weight-decay 0.1 "
            "--adam-beta1 0.9 "
            "--adam-beta2 0.98 "
        )

        # No --rollout-num-gpus / --rollout-num-gpus-per-engine: those are
        # inferred from /server_info on each external engine (1 prefill +
        # 1 decode + 2 regular, all tp=1).
        external_args = (
            "--rollout-external-engine-addrs "
            f"{EXTERNAL_HOST}:{PREFILL_PORT} "
            f"{EXTERNAL_HOST}:{DECODE_PORT} " + " ".join(f"{EXTERNAL_HOST}:{port}" for port in REGULAR_PORTS) + " "
        )

        ci_args = "--ci-test "

        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--loss-mask-type qwen3_5 "
            "--actor-num-nodes 1 "
            f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
        )

        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{U.get_default_wandb_args(__file__)} "
            f"{perf_args} "
            f"{external_args} "
            f"{ci_args} "
            f"{misc_args} "
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_TRAIN_GPUS,
            megatron_model_type=MODEL_TYPE,
            before_ray_job_submit=launch_external_engines,
        )
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
                p.wait()
        U.exec_command("pkill -9 sglang; true")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
