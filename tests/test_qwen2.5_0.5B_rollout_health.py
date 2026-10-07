"""Real training after a server/actor crash with a stale router registration."""

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path

import ray

import slime.utils.external_utils.command_utils as U
from slime.ray.training_recovery import RECOVERY_NAMESPACE, training_session_name

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute(crash_mode, directory=None):
    directory = Path(directory or tempfile.mkdtemp(prefix="slime_rollout_health_"))
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("test_qwen2.5_0.5B_training_recovery.py")
    spec = importlib.util.spec_from_file_location("recovery_e2e", source)
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
    arguments = recovery.train_args(
        directory, "debug", 1, f"/root/models/{MODEL_NAME}", "/root/datasets/gsm8k/train.parquet", 3, False
    )
    arguments = arguments.replace("--num-rollout 3", "--num-rollout 2")
    arguments = arguments.replace("training_recovery_test_helpers", "rollout_health_test_helpers")
    arguments += (
        "--rollout-function-path rollout_health_test_helpers.generate_rollout "
        "--over-sampling-batch-size 8 --rollout-health-check-interval 600 "
        "--rollout-health-check-timeout 5 "
        f"--rollout-session-id {directory.name} "
    )
    try:
        U.execute_train(
            train_args=arguments,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
            extra_env_vars={
                "PYTHONPATH": f"{U.repo_base_dir}/tests:{U.repo_base_dir}:/root/Megatron-LM",
                "RAY_ADDRESS": os.environ.get("SLIME_TEST_RAY_ADDRESS", "127.0.0.1:6379"),
                "SLIME_HEALTH_CRASH_MODE": crash_mode,
                "SLIME_HEALTH_TEST_DIR": str(directory),
                "NO_PROXY": "*",
                "no_proxy": "*",
                **{
                    key: ""
                    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
                },
            },
        )
        crash = json.loads((directory / "crash.json").read_text())
        assert crash["drain_seconds"] < 60
        assert json.loads((directory / "trained_0.json").read_text())["live_engines"] == 1
        assert json.loads((directory / "trained_1.json").read_text())["live_engines"] == 2
        assert (directory / "checkpoint/iter_0000001").exists()
        print(f"Rollout health verified: crash={crash_mode}, drain={crash['drain_seconds']:.2f}s, training completed.")
    finally:
        ray.init(address=os.environ.get("SLIME_TEST_RAY_ADDRESS", "127.0.0.1:6379"), namespace=RECOVERY_NAMESPACE)
        name = training_session_name(argparse.Namespace(rollout_session_id=directory.name))
        for suffix in ("", ":serving"):
            try:
                retained = ray.get_actor(name + suffix, namespace=RECOVERY_NAMESPACE)
            except ValueError:
                continue
            try:
                ray.get(retained.dispose.remote(), timeout=60)
            finally:
                ray.kill(retained, no_restart=True)
        ray.shutdown()
        if os.environ.get("SLIME_SCRIPT_EXTERNAL_RAY") != "1":
            subprocess.run(["ray", "stop", "--force"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash-mode", choices=["server", "actor"], default="server")
    parser.add_argument("--directory")
    parser.add_argument("--no-prepare", action="store_true")
    args = parser.parse_args()
    if not args.no_prepare:
        prepare()
    execute(args.crash_mode, args.directory)
