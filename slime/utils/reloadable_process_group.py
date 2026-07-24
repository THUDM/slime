import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import PrefixStore, _get_default_group, _get_default_store

from slime.utils import accelerator
from slime.utils.distributed_utils import get_gloo_group, init_gloo_group, set_gloo_group
from slime.utils.memory_utils import available_memory, clear_memory, print_memory

logger = logging.getLogger(__name__)

old_new_group_dict = {}
default_process_group_states = {}


@dataclass
class _DefaultProcessGroupState:
    backend: str
    timeout: timedelta
    store: Any
    rank: int
    world_size: int
    generation: int = 0
    accelerator_world_destroyed: bool = False


def register_default_process_group(timeout: timedelta) -> None:
    """Register the accelerator WORLD group so it can be destroyed and rebuilt.

    Keeping a reference to the rendezvous store is intentional.  It keeps the
    rank-0 TCPStore alive after ``destroy_process_group()`` and lets every
    generation use a fresh PrefixStore namespace, avoiding stale rendezvous
    keys when WORLD is recreated repeatedly.
    """
    if not dist.is_initialized():
        raise RuntimeError("Cannot register WORLD before torch.distributed is initialized")

    pid = os.getpid()
    backend = str(dist.get_backend())
    state = _DefaultProcessGroupState(
        backend=backend,
        timeout=timeout,
        store=_get_default_store(),
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )
    default_process_group_states[pid] = state
    logger.info(
        "Registered default WORLD process group for reload: backend=%s, rank=%s, world_size=%s",
        backend,
        state.rank,
        state.world_size,
    )


def _uses_accelerator_backend(backend: str) -> bool:
    backend = backend.lower()
    return "nccl" in backend or "mccl" in backend


def _new_default_process_group(state: _DefaultProcessGroupState, backend: str) -> None:
    state.generation += 1
    store = PrefixStore(f"slime-reloadable-world-{state.generation}-{backend}", state.store)
    dist.init_process_group(
        backend=backend,
        store=store,
        rank=state.rank,
        world_size=state.world_size,
        timeout=state.timeout,
    )


def _destroy_default_accelerator_process_group() -> None:
    state = default_process_group_states.get(os.getpid())
    if state is None or state.accelerator_world_destroyed or not _uses_accelerator_backend(state.backend):
        return

    # Use the CPU group as an out-of-band synchronization point.  PyTorch
    # requires all ranks to coordinate WORLD destruction/reinitialization.
    dist.barrier(group=get_gloo_group())
    dist.destroy_process_group()
    set_gloo_group(None)

    _new_default_process_group(state, backend="gloo")
    set_gloo_group(_get_default_group())
    state.accelerator_world_destroyed = True
    logger.info(
        "Destroyed default %s WORLD process group and initialized a temporary Gloo WORLD (generation %s)",
        state.backend,
        state.generation,
    )


def _reload_default_process_group() -> None:
    state = default_process_group_states.get(os.getpid())
    if state is None or not state.accelerator_world_destroyed:
        return

    # WORLD uses Gloo while the accelerator WORLD is destroyed, so this barrier
    # does not recreate an accelerator communicator before all ranks are ready.
    dist.barrier()
    dist.destroy_process_group()
    set_gloo_group(None)

    _new_default_process_group(state, backend=state.backend)
    init_gloo_group()
    state.accelerator_world_destroyed = False
    logger.info(
        "Reloaded default WORLD process group with backend %s (generation %s)",
        state.backend,
        state.generation,
    )


_COMM_MEMORY_CHECK_SKIP_OPS = {
    "all_gather_into_tensor",
    "allgather_into_tensor_coalesced",
    "barrier",
    "broadcast_object_list",
    "reduce_scatter_tensor",
    "all_to_all_single",
    "isend",
    "irecv",
}


def _should_check_memory_for_comm(op_name):
    return op_name not in _COMM_MEMORY_CHECK_SKIP_OPS


def monkey_patch_torch_dist():
    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        explicit_backend = args[2] if len(args) >= 3 else kwargs.get("backend")
        backend = str(explicit_backend) if explicit_backend is not None else str(dist.get_backend())
        normalized_backend = accelerator.process_group_backend(backend)
        if normalized_backend != backend:
            if len(args) >= 3:
                args = (*args[:2], normalized_backend, *args[3:])
            else:
                kwargs = {**kwargs, "backend": normalized_backend}
            backend = normalized_backend

        group = old_new_group(*args, **kwargs)

        # Before WORLD is registered, preserve the historical behavior of
        # leaving CPU groups and singleton groups untouched.  Afterwards every
        # cached subgroup must be reloadable because destroying WORLD
        # invalidates all of them, including Gloo and singleton groups.
        if backend == "gloo" and pid not in default_process_group_states:
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))

        # Historically singleton groups were left unwrapped because they do
        # not own a useful communicator.  Once WORLD itself is destroyed,
        # however, PyTorch invalidates *every* registered subgroup, including
        # singleton groups cached by Megatron.  Wrap them for actors that have
        # registered a reloadable WORLD so those cached references remain
        # usable after wake-up.  Preserve the old behavior for other callers.
        if len(ranks) == 1 and pid not in default_process_group_states:
            return group

        group = ReloadableProcessGroup(
            group,
            ranks,
            creation_args=args,
            creation_kwargs=kwargs,
            backend=backend,
        )
        return group

    dist.new_group = new_group

    def get_new_query_function(func):
        """Wrap query functions (get_rank, get_world_size, etc.) without memory check."""

        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    def get_new_comm_function(func, op_name=None):
        """Wrap communication functions with memory check."""

        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            check_memory = True if op_name is None else _should_check_memory_for_comm(op_name)
            with _wrap_low_level_call(check_memory=check_memory):
                return func(*args, **kwargs)

        return new_function

    dist.get_rank = get_new_query_function(dist.get_rank)
    dist.get_world_size = get_new_query_function(dist.get_world_size)
    dist.get_backend = get_new_query_function(dist.get_backend)
    dist.get_global_rank = get_new_query_function(dist.get_global_rank)
    dist.get_group_rank = get_new_query_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_query_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_comm_function(dist.all_reduce)
    dist.all_gather = get_new_comm_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_comm_function(dist.all_gather_into_tensor, "all_gather_into_tensor")
    dist.all_gather_object = get_new_comm_function(dist.all_gather_object)
    dist.all_to_all = get_new_comm_function(dist.all_to_all)
    dist.all_to_all_single = get_new_comm_function(dist.all_to_all_single, "all_to_all_single")
    dist.broadcast = get_new_comm_function(dist.broadcast)
    dist.broadcast_object_list = get_new_comm_function(dist.broadcast_object_list, "broadcast_object_list")
    dist.reduce = get_new_comm_function(dist.reduce)
    dist.reduce_scatter = get_new_comm_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_comm_function(dist.reduce_scatter_tensor, "reduce_scatter_tensor")
    dist.scatter = get_new_comm_function(dist.scatter)
    dist.scatter_object_list = get_new_comm_function(dist.scatter_object_list)
    dist.gather = get_new_comm_function(dist.gather)
    dist.gather_object = get_new_comm_function(dist.gather_object)
    dist.barrier = get_new_comm_function(dist.barrier, "barrier")
    dist.send = get_new_comm_function(dist.send)
    dist.send_object_list = get_new_comm_function(dist.send_object_list)
    dist.recv = get_new_comm_function(dist.recv)
    dist.recv_object_list = get_new_comm_function(dist.recv_object_list)
    dist._coalescing_manager = get_new_comm_function(dist._coalescing_manager)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_comm_function(dist.isend, "isend")
    dist.irecv = get_new_comm_function(dist.irecv, "irecv")

    def get_new_p2pop_function(func):
        def new_function(*args, **kwargs):
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg

            args = (convert(arg) for arg in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    GROUPS = {}

    def __init__(self, group, ranks, *, creation_args=(), creation_kwargs=None, backend="nccl"):
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.group_info = {
            "ranks": ranks,
            "args": tuple(creation_args),
            "kwargs": dict(creation_kwargs or {}),
            "backend": backend,
        }
        pid = os.getpid()
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def __getattr__(self, name):
        return getattr(self.group, name)

    @staticmethod
    def destroy_process_groups():
        pid = os.getpid()
        for reloadable_group in ReloadableProcessGroup.GROUPS.get(pid, []):
            if reloadable_group.group is None:
                continue
            try:
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            del reloadable_group.group
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups():
        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        backend_counts = {}
        for reloadable_group in reloadable_groups:
            backend = reloadable_group.group_info["backend"]
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        logger.info(
            "Reloading %s process groups in pid %s: %s",
            len(reloadable_groups),
            pid,
            backend_counts,
        )
        old_new_group = old_new_group_dict.get(pid)
        for reloadable_group in reloadable_groups:
            if reloadable_group.group is not None:
                continue
            group = old_new_group(
                *reloadable_group.group_info["args"],
                **reloadable_group.group_info["kwargs"],
            )
            reloadable_group.group = group

    def rank(self) -> int:
        return self.group.rank()

    def size(self) -> int:
        return self.group.size()

    def name(self) -> str:
        return self.group.name()

    def shutdown(self) -> None:
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call(check_memory=_should_check_memory_for_comm(method)):
            return getattr(inner, method)(*args, **kwargs)

    def _fwd_query(self, method, *args, **kwargs):
        """Forward non-communication calls without memory check."""
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        return getattr(inner, method)(*args, **kwargs)

    def barrier(self, *a, **kw):
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, *a, **kw):
        return self._fwd("allgather_into_tensor_coalesced", *a, **kw)

    def gather(self, *a, **kw):
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, *a, **kw):
        return self._fwd("reduce_scatter_tensor_coalesced", *a, **kw)

    def alltoall_base(self, *a, **kw):
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        return self._fwd("recv_anysource", *a, **kw)

    def _start_coalescing(self, *a, **kw):
        return self._fwd_query("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd_query("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd_query("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd_query("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


def destroy_process_groups():
    """Destroy registered subgroups and replace accelerator WORLD with a temporary Gloo WORLD."""
    state = default_process_group_states.get(os.getpid())
    if state is not None and not state.accelerator_world_destroyed and _uses_accelerator_backend(state.backend):
        dist.barrier(group=get_gloo_group())
    ReloadableProcessGroup.destroy_process_groups()
    _destroy_default_accelerator_process_group()


def reload_process_groups():
    """Restore accelerator WORLD and recreate all registered subgroups."""
    _reload_default_process_group()
    ReloadableProcessGroup.reload_process_groups()


@contextmanager
def _wrap_low_level_call(check_memory=True):
    try:
        if check_memory:
            mem_info = available_memory()
            if mem_info["free_GB"] < 3:
                clear_memory()
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        e.add_note(f"{mem_info=}")
        raise
