import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

from slime.utils.misc import exec_command

_ = exec_command

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[1]


def convert_checkpoint(model_name, model_type, num_gpus: int, dir_dst="/root"):
    # TODO shall we make it in host-mapped folder and thus can cache it to speedup CI
    path_dst = f"{dir_dst}/{model_name}_torch_dist"
    if Path(path_dst).exists():
        print(f"convert_checkpoint skip {path_dst} since exists")
        return

    exec_command(
        f"source {repo_base_dir}/scripts/models/{model_type}.sh && "
        f"PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node {num_gpus} tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint /root/models/{model_name} "
        f"--save {path_dst}"
    )


def hf_download_dataset(full_name: str):
    _, partial_name = full_name.split("/")
    exec_command(f"hf download --repo-type dataset {full_name} --local-dir /root/datasets/{partial_name}")


def execute_train(
    train_args: str,
    num_gpus: int,
    model_type: Optional[str],
    train_script: str = "train.py",
    before_ray_job_submit=None,
    extra_env_vars={},
):
    external_ray = bool(int(os.environ.get("SLIME_SCRIPT_EXTERNAL_RAY", "0")))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

    exec_command(
        "pkill -9 sglang; "
        "sleep 3; "
        f"{'' if external_ray else 'ray stop --force; '}"
        f"{'' if external_ray else 'pkill -9 ray; '}"
        # cannot be run in CI, o/w kill the parent script
        # TODO: do we really need this kill? (or can we instead kill slime)
        # "pkill -9 python; "
        "pkill -9 slime; "
        "sleep 3; "
        f"{'' if external_ray else 'pkill -9 ray; '}"
        # "pkill -9 python; "
        "pkill -9 slime; "
        "pkill -9 redis; "
        "true; "
    )

    if not external_ray:
        exec_command(
            # will prevent ray from buffering stdout/stderr
            f"export PYTHONBUFFERED=16 && "
            f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus} --disable-usage-stats"
        )

    if (f := before_ray_job_submit) is not None:
        f()

    runtime_env_json = json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "/root/Megatron-LM/",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": str(int(check_has_nvlink())),
                "no_proxy": f"127.0.0.1,{master_addr}",
                # This is needed by megatron / torch distributed in multi-node setup
                "MASTER_ADDR": master_addr,
                **(
                    {
                        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
                        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
                        "CUDA_COREDUMP_GENERATION_FLAGS": "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory",
                        "CUDA_COREDUMP_FILE": "/tmp/cuda_coredump_%h.%p.%t",
                    }
                    if get_bool_env_var("SLIME_SCRIPT_ENABLE_CUDA_CORE_DUMP")
                    else {}
                ),
                **extra_env_vars,
            }
        }
    )

    source_cmd = f'source "{repo_base_dir}/scripts/models/{model_type}.sh" && ' if model_type is not None else ""
    model_args_str = "${MODEL_ARGS[@]}" if model_type is not None else ""

    if bool(int(os.environ.get("SLIME_SCRIPT_ENABLE_RAY_SUBMIT", "1"))):
        exec_command(
            f"export PYTHONBUFFERED=16 && "
            f"{source_cmd}"
            # TODO should this 127.0.0.1 be `master_addr` instead
            f'ray job submit --address="http://127.0.0.1:8265" '
            f"--runtime-env-json='{runtime_env_json}' "
            f"-- python3 {train_script} "
            f"{model_args_str} "
            f"{train_args}"
        )


def check_has_nvlink():
    output = exec_command("nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True)
    return int(output) > 0


def get_default_wandb_args(test_file: str, run_name_prefix: Optional[str] = None, run_id: Optional[str] = None):
    if not os.environ.get("WANDB_API_KEY"):
        print("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""

    test_file = Path(test_file)
    test_name = test_file.stem
    if len(test_name) < 6:
        test_name = f"{test_file.parent.name}_{test_name}"

    wandb_run_name = run_id or create_run_id()
    if (x := os.environ.get("GITHUB_COMMIT_NAME")) is not None:
        wandb_run_name += f"_{x}"
    if (x := run_name_prefix) is not None:
        wandb_run_name = f"{x}_{wandb_run_name}"

    # do not put wandb_api_key value here to avoid leaking to logs explicitly
    return (
        "--use-wandb "
        f"--wandb-project slime-ci-{test_name} "
        f"--wandb-group {wandb_run_name} "
        f"--wandb-key ${{WANDB_API_KEY}} "
        "--disable-wandb-random-suffix "
    )


def create_run_id() -> str:
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S") + f"-{random.Random().randint(0, 999):03d}"


_warned_bool_env_var_keys = set()


# copied from SGLang
def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            print(f"get_bool_env_var({name}) see non-understandable value={value} and treat as false")
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def get_env_enable_infinite_run():
    return get_bool_env_var("SLIME_TEST_ENABLE_INFINITE_RUN", "false")


def save_to_temp_file(text: str, ext: str):
    path = Path(f"/tmp/slime_temp_file_{time.time()}_{random.randrange(0, 10000000)}.{ext}")
    path.write_text(text)
    print(f"Write the following content to {path=}: {text=}")
    return str(path)
