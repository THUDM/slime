import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from slime.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)

old_new_group_dict = {}


def monkey_patch_torch_dist():
    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    setattr(dist, "old_new_group", old_new_group)

    def new_group(*args, **kwargs):
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))

        if len(ranks) == 1:
            return group

        group = ReloadableProcessGroup(group, ranks)
        return group

    dist.new_group = new_group

    def get_new_function(func):
        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            with _wrap_low_level_call():
                return func(*args, **kwargs)

        return new_function

    dist.get_rank = get_new_function(dist.get_rank)
    dist.get_world_size = get_new_function(dist.get_world_size)
    dist.get_backend = get_new_function(dist.get_backend)
    dist.get_global_rank = get_new_function(dist.get_global_rank)
    dist.get_group_rank = get_new_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_function(dist.all_reduce)
    dist.all_gather = get_new_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_function(dist.all_gather_into_tensor)
    dist.all_gather_object = get_new_function(dist.all_gather_object)
    dist.all_to_all = get_new_function(dist.all_to_all)
    dist.all_to_all_single = get_new_function(dist.all_to_all_single)
    dist.broadcast = get_new_function(dist.broadcast)
    dist.reduce = get_new_function(dist.reduce)
    dist.reduce_scatter = get_new_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_function(dist.reduce_scatter_tensor)
    dist.scatter = get_new_function(dist.scatter)
    dist.gather = get_new_function(dist.gather)
    dist.barrier = get_new_function(dist.barrier)
    dist.send = get_new_function(dist.send)
    dist.recv = get_new_function(dist.recv)
    dist._coalescing_manager = get_new_function(dist._coalescing_manager)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_function(dist.isend)
    dist.irecv = get_new_function(dist.irecv)

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

    def __init__(self, group, ranks):
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.group_info = {
            "ranks": ranks,
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
        for reloadable_group in ReloadableProcessGroup.GROUPS[pid]:
            if reloadable_group.group is None:
                continue
            dist.destroy_process_group(reloadable_group.group)

            del reloadable_group.group
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups():
        pid = os.getpid()
        logger.info(f"Reloading {len(ReloadableProcessGroup.GROUPS[pid])} process groups in pid {pid}")
        old_new_group = old_new_group_dict[pid]
        for reloadable_group in ReloadableProcessGroup.GROUPS[pid]:
            if reloadable_group.group is not None:
                continue
            group = old_new_group(ranks=reloadable_group.group_info["ranks"], backend="nccl")
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
        with _wrap_low_level_call():
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
        return self._fwd("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


def destroy_process_groups():
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups():
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups()


@contextmanager
def _wrap_low_level_call():
    try:
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        e.add_note(f"{mem_info=}")
        raise
