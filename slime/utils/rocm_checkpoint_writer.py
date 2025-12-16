import torch
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync


class ROCmFileSystemWriterAsync(FileSystemWriterAsync):
    """
    FileSystemWriterAsync wrapper for ROCm compatibility.

    On ROCm/HIP, using non_blocking=True causes tensors to be stored in pinned memory,
    which triggers segmentation faults when forking subprocesses afterward.
    """

    @staticmethod
    def preload_tensors(*args, **kwargs):
        # Change argument non_blocking to False on HIP platform
        # The tensors will be stored in pinned memory if non_blocking=True
        # Currently on the ROCm platform, forking a subprocess afterward
        # with pinned_memory=True will trigger segmentation fault
        if torch.version.hip:
            print("HIP/ROCm detected: setting non_blocking=False in preload_tensors")
            if "non_blocking" in kwargs:
                kwargs["non_blocking"] = False
            elif len(args) > 1 and isinstance(args[-1], bool):
                # non_blocking is typically the last argument
                args = args[:-1] + (False,)

        return FileSystemWriterAsync.preload_tensors(*args, **kwargs)
