import os
from contextlib import contextmanager
import pytest
import ray

CKPT_ARGS=[
   "--hf-checkpoint", "/root/Qwen3-0.6B",
]


ROLLOUT_ARGS=[
    "--prompt-data", "/root/dapo-math-17k/dapo-math-17k.jsonl",
    "--input-key", "prompt",
    "--label-key", "label",
    "--apply-chat-template",
    "--rollout-shuffle",
    "--rm-type", "deepscaler",
    "--num-rollout", "3000",
    "--rollout-batch-size", "16",
    "--n-samples-per-prompt", "16",
    "--rollout-max-response-len", "8192",
    "--rollout-temperature", "0.8",
    "--global-batch-size", "128",
]

GRPO_ARGS=[
    "--advantage-estimator", "grpo",
    "--kl-loss-coef", "0.00",
    "--kl-loss-type", "low_var_kl",
    "--kl-coef", "0.00",
    "--entropy-coef", "0.00",
    "--eps-clip", "0.2",
    "--eps-clip-high", "0.28",
]

OPTIMIZER_ARGS=[
    "--optimizer", "adam",
    "--lr", "1e-6",
    "--lr-decay-style", "constant",
    "--weight-decay", "0.1",
    "--adam-beta1", "0.9",
    "--adam-beta2", "0.98",
]

SGLANG_ARGS=[
    "--rollout-num-gpus-per-engine", "1",
    "--load-format", "dummy",
]

TRAIN_ARGS=[
    "--actor-num-nodes", "1",
    "--actor-num-gpus-per-node", "4",
]

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    os.environ["PYTHONBUFFERED"] = "16"
    # TODO: Temporary workaround to assign environment variables both outside and inside Ray.
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    os.environ["SLIME_BACKEND"] = "fsdp"

# Ref: https://github.com/ray-project/ray/blob/ray-2.49.1/python/ray/tests/conftest.py
def get_default_fixure_system_config():
    system_config = {
        "object_timeout_milliseconds": 200,
        "health_check_initial_delay_ms": 0,
        "health_check_failure_threshold": 10,
        "object_store_full_delay_ms": 100,
        "local_gc_min_interval_s": 1,
    }
    return system_config

def get_default_fixture_ray_kwargs():
    system_config = get_default_fixure_system_config()
    ray_kwargs = {
        "num_cpus": 1,
        "object_store_memory": 150 * 1024 * 1024,
        "dashboard_port": None,
        "namespace": "default_test_namespace",
        "_system_config": system_config,
    }
    return ray_kwargs

def get_slime_ray_kwargs():
    ray_kwargs = {
        "runtime_env": {
            "env_vars": {
                "no_proxy": "localhost,127.0.0.1,0.0.0.0",
                "SLIME_BACKEND": "fsdp",
            }
        }
    }
    return ray_kwargs

@contextmanager
def _ray_start(**kwargs):
    init_kwargs = get_default_fixture_ray_kwargs()
    init_kwargs.update(kwargs)
    init_kwargs.update(get_slime_ray_kwargs())
    # Start the Ray processes.
    address_info = ray.init("local", **init_kwargs)

    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()
    # Delete the cluster address just in case.
    ray._common.utils.reset_ray_address()


@pytest.fixture
def ray_start_regular(request):
    param = getattr(request, "param", {})
    with _ray_start(**param) as res:
        yield res

@pytest.fixture
def ray_start_4_gpus_unlimited_cpus(request):
    param = getattr(request, "param", {})
    with _ray_start(num_cpus=None, num_gpus=4, object_store_memory=None, **param) as res:
        yield res