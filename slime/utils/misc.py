import importlib
import subprocess
import warnings
from collections import defaultdict
from collections.abc import Iterable
from functools import cache
from typing import Any

import torch

from slime.utils.http_utils import is_port_available


def decode_meta_array(
    meta_info: dict[str, Any],
    keys: str | Iterable[str],
    *,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        if key in meta_info:
            value = meta_info[key]
            break
    else:
        return None

    if value is None:
        return None
    if isinstance(value, str):
        import pybase64

        value = pybase64.b64decode(value.encode("ascii"))
    if isinstance(value, bytes | bytearray | memoryview):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable", category=UserWarning)
            return torch.frombuffer(value, dtype=dtype)
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=dtype).reshape(-1)
    if hasattr(value, "flags") and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, dtype=dtype).reshape(-1)


def decode_int32_meta_array(meta_info: dict[str, Any], keys: str | Iterable[str]) -> torch.Tensor | None:
    return decode_meta_array(meta_info, keys, dtype=torch.int32)


@cache
def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


class SingletonMeta(type):
    """
    A metaclass for creating singleton classes.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def clear_instances(cls):
        cls._instances = {}


def exec_command(cmd: str, capture_output: bool = False) -> str | None:
    print(f"EXEC: {cmd}", flush=True)

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            shell=False,
            check=True,
            capture_output=capture_output,
            **(dict(text=True) if capture_output else {}),
        )
    except subprocess.CalledProcessError as e:
        if capture_output:
            print(f"{e.stdout=} {e.stderr=}")
        raise

    if capture_output:
        print(f"Captured stdout={result.stdout} stderr={result.stderr}")
        return result.stdout


def get_current_node_ip():
    # Lazy import so CPU-only code paths (rm_hub scoring, plugin contracts,
    # etc.) can use other helpers in this module without requiring ray.
    import ray

    address = ray._private.services.get_node_ip_address()
    # strip ipv6 address
    address = address.strip("[]")
    return address


def get_free_port(start_port=10000, consecutive=1):
    # find the port where port, port + 1, port + 2, ... port + consecutive - 1 are all available
    port = start_port
    while not all(is_port_available(port + i) for i in range(consecutive)):
        port += 1
    return port


def compute_rollout_schedule(dataset_size: int, rollout_batch_size: int, num_epoch: int) -> tuple[int, frozenset[int]]:
    if dataset_size < 0 or rollout_batch_size <= 0 or num_epoch <= 0:
        raise ValueError("dataset_size must be non-negative and rollout_batch_size and num_epoch must be positive")

    num_rollout = num_epoch * dataset_size // rollout_batch_size
    if dataset_size > 0:
        num_rollout = max(1, num_rollout)

    epoch_boundaries = frozenset(
        rollout_id
        for rollout_id in range(num_rollout)
        if (rollout_id + 1) * rollout_batch_size // dataset_size != rollout_id * rollout_batch_size // dataset_size
    )
    return num_rollout, epoch_boundaries


def should_run_periodic_action(
    rollout_id: int,
    interval: int | None,
    epoch_boundaries: frozenset[int] | None = None,
    num_rollout: int | None = None,
) -> bool:
    """
    Return True when a periodic action (eval/save/checkpoint) should run.

    Args:
        rollout_id: The current rollout index (0-based).
        interval: Desired cadence; disables checks when None.
        epoch_boundaries: Rollout indexes at which the data source enters a new epoch.
    """
    if interval is None:
        return False

    if num_rollout is not None and rollout_id == num_rollout - 1:
        return True

    step = rollout_id + 1
    return (step % interval == 0) or (epoch_boundaries is not None and rollout_id in epoch_boundaries)


class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner


# details: https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby
def group_by(iterable, key=None):
    """Similar to itertools.groupby, but do not require iterable to be sorted"""
    ret = defaultdict(list)
    for item in iterable:
        ret[key(item) if key is not None else item].append(item)
    return dict(ret)
