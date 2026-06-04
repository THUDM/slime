"""E2E test for --rollout-external-engine-addrs with a pure-PD external fleet.

Spawns four SGLang servers out-of-band on a single 8-GPU box (all tp=1):
- 2 prefill (``--disaggregation-mode prefill``, mooncake transfer backend)
- 2 decode  (``--disaggregation-mode decode``,  mooncake transfer backend)

and points slime at all four via ``--rollout-external-engine-addrs ...``.
The remaining 4 GPUs train. slime queries ``/server_info`` on each engine to
infer per-engine TP / GPU counts and registers them to its PD-enabled router.

Weight sync uses ``--update-weight-mode delta --update-weight-transport disk``
so the post-train sync writes sparse safetensors to a shared dir and the
external engines load them via ``update_weights_from_disk(load_format=delta)``
— that's the only sync path that actually works for pre-launched workers (no
NCCL group between trainer and external engines).
"""

import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3.5-0.8B"
MODEL_TYPE = "qwen3.5-0.8B"
NUM_GPUS = 8
NUM_TRAIN_GPUS = 4
NUM_PREFILL_ENGINES = 2
NUM_DECODE_ENGINES = 2

EXTERNAL_HOST = "127.0.0.1"
PREFILL_PORTS = [13150, 13151]
DECODE_PORTS = [13152, 13153]
BOOTSTRAP_PORTS = [13160, 13161]


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
    """Partition the 8 visible GPUs: 4 train + 2 prefill + 2 decode."""
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(NUM_GPUS))).split(",")
    assert len(all_gpus) >= NUM_GPUS, f"Expected at least {NUM_GPUS} GPUs, got {len(all_gpus)}"
    train_gpus = all_gpus[:NUM_TRAIN_GPUS]
    cursor = NUM_TRAIN_GPUS
    prefill_gpus = all_gpus[cursor : cursor + NUM_PREFILL_ENGINES]
    cursor += NUM_PREFILL_ENGINES
    decode_gpus = all_gpus[cursor : cursor + NUM_DECODE_ENGINES]
    return train_gpus, prefill_gpus, decode_gpus


def _launch_sglang_server(
    *,
    gpus: list[str],
    port: int,
    tp: int,
    log_path: str,
    disaggregation_mode: str,
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
        "--disaggregation-mode",
        disaggregation_mode,
        "--disaggregation-transfer-backend",
        "mooncake",
    ]
    if disaggregation_bootstrap_port is not None:
        cmd += ["--disaggregation-bootstrap-port", str(disaggregation_bootstrap_port)]

    log_file = open(log_path, "w")
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    print(
        f"Starting external sglang {disaggregation_mode} server on GPUs {gpus} "
        f"port={port} tp={tp} (pid={process.pid}), log: {log_path}"
    )

    # Wait up to ~10 minutes for /server_info to come up.  /health_generate
    # is unreliable for prefill/decode-only nodes, so we poll /server_info
    # — that's what slime's discover_external_engines uses anyway.
    deadline = time.time() + 600
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{disaggregation_mode} server exited with code {process.returncode}; check {log_path}")
        try:
            req = urllib.request.urlopen(f"http://{EXTERNAL_HOST}:{port}/server_info", timeout=2)
            if req.status == 200:
                print(f"External sglang {disaggregation_mode} server is ready on GPUs {gpus}")
                return process
        except Exception:
            pass
        time.sleep(5)

    process.kill()
    raise RuntimeError(f"{disaggregation_mode} server failed to start within timeout; check {log_path}")


def execute():
    train_gpus, prefill_gpus, decode_gpus = _get_gpu_split()
    processes: list[subprocess.Popen] = []

    # Restrict CUDA_VISIBLE_DEVICES to training GPUs before Ray starts so
    # ray's bundle allocator doesn't try to claim the external sglang GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)

    def launch_external_engines():
        for idx, (gpu, port, bootstrap_port) in enumerate(
            zip(prefill_gpus, PREFILL_PORTS, BOOTSTRAP_PORTS, strict=True)
        ):
            processes.append(
                _launch_sglang_server(
                    gpus=[gpu],
                    port=port,
                    tp=1,
                    disaggregation_mode="prefill",
                    disaggregation_bootstrap_port=bootstrap_port,
                    log_path=f"/tmp/sglang_external_prefill_{idx}.log",
                )
            )
        for idx, (gpu, port) in enumerate(zip(decode_gpus, DECODE_PORTS, strict=True)):
            processes.append(
                _launch_sglang_server(
                    gpus=[gpu],
                    port=port,
                    tp=1,
                    disaggregation_mode="decode",
                    log_path=f"/tmp/sglang_external_decode_{idx}.log",
                )
            )

    delta_dir_cm = tempfile.TemporaryDirectory(prefix="slime_external_pd_delta_")
    delta_dir = delta_dir_cm.name
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
            # Nonzero entropy coef guarantees a nonzero gradient even when all
            # rewards in a group tie (advantages=0), so the delta sync writes
            # real sparse files instead of an empty no-op.
            "--entropy-coef 0.01 "
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
        # inferred from /server_info on each external engine (2 prefill +
        # 2 decode, all tp=1).
        all_addrs = [f"{EXTERNAL_HOST}:{port}" for port in (*PREFILL_PORTS, *DECODE_PORTS)]
        external_args = "--rollout-external-engine-addrs " + " ".join(all_addrs) + " "

        # External engines have no NCCL group with the trainer, so weight
        # updates have to go through the disk-backed delta path: the trainer
        # writes sparse safetensors per sync, the engines pull via
        # update_weights_from_disk(load_format="delta", files=...).
        delta_args = (
            "--update-weight-mode delta "
            "--update-weight-transport disk "
            "--update-weight-encoding deltas "
            f"--update-weight-delta-dir {delta_dir} "
            "--update-weight-delta-keep-files "
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
            f"{delta_args} "
            f"{ci_args} "
            f"{misc_args} "
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_TRAIN_GPUS,
            megatron_model_type=MODEL_TYPE,
            before_ray_job_submit=launch_external_engines,
        )

        delta_files = list(Path(delta_dir).glob("weight_v*/*.safetensors"))
        assert delta_files, f"No disk delta safetensors were written under {delta_dir}"
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
                p.wait()
        U.exec_command("pkill -9 sglang; true")
        delta_dir_cm.cleanup()


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
