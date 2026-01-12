"""
This file is not for slime framework itself, but as an optional utility to easily launch slime jobs and tests.
"""

import datetime
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from slime.utils.misc import exec_command
from slime.utils.typer_utils import dataclass_cli

_ = exec_command, dataclass_cli

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[3]


def convert_checkpoint(
    model_name,
    megatron_model_type,
    num_gpus_per_node: int,
    multinode: bool = False,
    extra_args: str = "",
    dir_dst: str = "/root",
    hf_checkpoint: str | None = None,
):
    hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"

    # TODO shall we make it in host-mapped folder and thus can cache it to speedup CI
    path_dst = f"{dir_dst}/{model_name}_torch_dist"
    if Path(path_dst).exists():
        print(f"convert_checkpoint skip {path_dst} since exists")
        return

    multinode_args = ""
    if multinode:
        # This variable can be provided via:
        # `export SLURM_JOB_HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")`
        print(f"{os.environ.get('SLURM_JOB_HOSTNAMES')=} {os.environ.get('SLURM_NODEID')=}")
        job_hostnames = os.environ["SLURM_JOB_HOSTNAMES"].strip().split("\n")
        master_addr = job_hostnames[0]
        nnodes = len(job_hostnames)
        node_rank = int(os.environ["SLURM_NODEID"])

        multinode_args = (
            f"--master-addr {master_addr} " "--master-port 23456 " f"--nnodes={nnodes} " f"--node-rank {node_rank} "
        )

    exec_command(
        f"source {repo_base_dir}/scripts/models/{megatron_model_type}.sh && "
        f"PYTHONPATH=/root/Megatron-LM "
        f"torchrun "
        f"--nproc-per-node {num_gpus_per_node} "
        f"{multinode_args}"
        f"tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint {hf_checkpoint} "
        f"--save {path_dst}"
        f"{extra_args}"
    )


def rsync_simple(path_src: str, path_dst: str):
    exec_command(f"mkdir -p {path_dst} && rsync -a --info=progress2 {path_src}/ {path_dst}")


def hf_download_dataset(full_name: str):
    _, partial_name = full_name.split("/")
    exec_command(f"hf download --repo-type dataset {full_name} --local-dir /root/datasets/{partial_name}")


def fp8_cast_bf16(path_src, path_dst):
    if Path(path_dst).exists():
        print(f"fp8_cast_bf16 skip {path_dst} since exists")
        return

    exec_command(
        "python tools/fp8_cast_bf16.py " f"--input-fp8-hf-path {path_src} " f"--output-bf16-hf-path {path_dst} "
    )


# This class can be extended by concrete scripts
@dataclass
class ExecuteTrainConfig:
    cuda_core_dump: bool = False
    num_nodes: int = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    extra_env_vars: str = ""


def execute_train(
    train_args: str,
    num_gpus_per_node: int,
    megatron_model_type: str | None,
    train_script: str = "train.py",
    before_ray_job_submit=None,
    extra_env_vars=None,
    config: ExecuteTrainConfig | None = None,
):
    if extra_env_vars is None:
        extra_env_vars = {}
    if config is None:
        config = ExecuteTrainConfig()
    external_ray = get_bool_env_var("SLIME_SCRIPT_EXTERNAL_RAY")
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

    train_backend_fsdp = "--train-backend fsdp" in train_args
    assert train_backend_fsdp == (megatron_model_type is None)

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
            f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus_per_node} --disable-usage-stats"
        )

    if (f := before_ray_job_submit) is not None:
        f()

    runtime_env_json = json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "/root/Megatron-LM/",
                # If setting this in FSDP, the computation communication overlapping may have issues
                **(
                    {}
                    if train_backend_fsdp
                    else {
                        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                    }
                ),
                "NCCL_NVLS_ENABLE": str(int(check_has_nvlink())),
                "no_proxy": f"127.0.0.1,{master_addr}",
                # This is needed by megatron / torch distributed in multi-node setup
                "MASTER_ADDR": master_addr,
                **(
                    {
                        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
                        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
                        "CUDA_COREDUMP_GENERATION_FLAGS": "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory",
                        "CUDA_COREDUMP_FILE": "/root/shared_data/cuda_coredump_%h.%p.%t",
                    }
                    if config.cuda_core_dump
                    else {}
                ),
                **extra_env_vars,
                **_parse_extra_env_vars(config.extra_env_vars),
            }
        }
    )

    if get_bool_env_var("SLIME_SCRIPT_ENABLE_RAY_SUBMIT", "1"):
        cmd_megatron_model_source = (
            f'source "{repo_base_dir}/scripts/models/{megatron_model_type}.sh" && '
            if megatron_model_type is not None
            else ""
        )
        exec_command(
            f"export no_proxy=127.0.0.1 && export PYTHONBUFFERED=16 && "
            f"{cmd_megatron_model_source}"
            f'ray job submit --address="http://127.0.0.1:8265" '
            f"--runtime-env-json='{runtime_env_json}' "
            f"-- python3 {train_script} "
            f"{'${MODEL_ARGS[@]}' if megatron_model_type is not None else ''} "
            f"{train_args}"
        )


def _parse_extra_env_vars(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {kv[0]: kv[1] for item in text.split(" ") if item.strip() != "" if (kv := item.split("=")) or True}


def check_has_nvlink():
    output = exec_command("nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True)
    return int(output) > 0


def get_default_wandb_args(test_file: str, run_name_prefix: str | None = None, run_id: str | None = None):
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

    # Use the actual key value from environment to avoid shell expansion issues
    wandb_key = os.environ.get("WANDB_API_KEY")
    return (
        "--use-wandb "
        f"--wandb-project slime-{test_name} "
        f"--wandb-group {wandb_run_name} "
        f"--wandb-key '{wandb_key}' "
        "--disable-wandb-random-suffix "
    )


def create_run_id() -> str:
    return datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S") + f"-{random.Random().randint(0, 999):03d}"


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


NUM_GPUS_OF_HARDWARE = {
    "H100": 8,
    "GB200": 4,
    "GB300": 4,
}

GENERATION_HARDWARE = {
    "H100": "Hopper",
    "GB200": "Blackwell",
    "GB300": "Blackwell",
}
