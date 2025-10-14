import datetime
import json
import os
import random
import subprocess
from pathlib import Path

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[1]


def convert_checkpoint(model_name, model_type):
    # TODO shall we make it in host-mapped folder and thus can cache it to speedup CI
    path_dst = f"/root/{model_name}_torch_dist"
    if Path(path_dst).exists():
        print(f"convert_checkpoint skip {path_dst} since exists")
        return

    exec_command(
        f"source {repo_base_dir}/scripts/models/{model_type}.sh && "
        "PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint /root/models/{model_name} "
        f"--save {path_dst}"
    )


def execute_train(
    train_args: str,
    num_gpus: int,
    model_type: str,
    master_addr: str = "127.0.0.1",
):
    exec_command(
        "pkill -9 sglang; "
        "sleep 3; "
        "ray stop --force; "
        "pkill -9 ray; "
        # cannot be run in CI, o/w kill the parent script
        # TODO: do we really need this kill? (or can we instead kill slime)
        # "pkill -9 python; "
        "pkill -9 slime; "
        "sleep 3; "
        "pkill -9 ray; "
        # "pkill -9 python; "
        "pkill -9 slime; "
        "pkill -9 redis; "
        "true; "
    )

    exec_command(
        # will prevent ray from buffering stdout/stderr
        f"export PYTHONBUFFERED=16 && "
        f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus} --disable-usage-stats"
    )

    runtime_env_json = json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "/root/Megatron-LM/",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": str(int(check_has_nvlink())),
                "no_proxy": f"127.0.0.1,{master_addr}",
            }
        }
    )

    exec_command(
        f"export no_proxy=127.0.0.1 && export PYTHONBUFFERED=16 && "
        f'source "{repo_base_dir}/scripts/models/{model_type}.sh" && '
        # TODO should this 127.0.0.1 be `master_addr` instead
        f'ray job submit --address="http://127.0.0.1:8265" '
        f"--runtime-env-json='{runtime_env_json}' "
        "-- python3 train.py "
        "${MODEL_ARGS[@]} "
        f"{train_args}"
    )


def check_has_nvlink():
    output = exec_command("nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True)
    return int(output) > 0


def get_default_wandb_args(test_file: str):
    if not os.environ.get("WANDB_API_KEY"):
        print("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""

    test_name = Path(test_file).stem

    run_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 1000000000)}"
    if (x := os.environ.get("GITHUB_COMMIT_NAME")) is not None:
        run_name += f"_{x}"

    # do not put wandb_api_key value here to avoid leaking to logs explicitly
    return (
        "--use-wandb "
        f"--wandb-project slime-ci-{test_name} "
        f"--wandb-group {run_name} "
        f"--wandb-key ${{WANDB_API_KEY}} "
    )


def exec_command(cmd: str, capture_output: bool = False):
    print(f"EXEC: {cmd}", flush=True)
    result = subprocess.run(["bash", "-c", cmd], shell=False, check=True, capture_output=capture_output)
    if capture_output:
        return result.stdout
