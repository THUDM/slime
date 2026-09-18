from __future__ import annotations

import errno
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class DiskTensorRef:
    """Small, Ray-serializable reference to a single-tensor safetensors file.

    Files are written and read with the official safetensors implementation.
    """

    path: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    kind: str | None = None
    validated: bool = False

    @classmethod
    def write(
        cls,
        tensor: torch.Tensor,
        path: str | Path,
        *,
        kind: str | None = None,
        validated: bool = False,
    ) -> DiskTensorRef:
        tensor = tensor.detach().cpu().contiguous()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"tensor": tensor}, path, metadata={"kind": kind} if kind else None)

        return cls(
            path=str(path),
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
            nbytes=tensor.numel() * tensor.element_size(),
            kind=kind,
            validated=validated,
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        dtype = getattr(torch, self.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported tensor dtype in disk reference: {self.dtype!r}")
        return dtype

    def link(self, path: str | Path) -> DiskTensorRef:
        """Give immutable tensor data another lifetime without loading it into RAM."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(self.path, path)
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
            # A different filesystem or one without hard links still needs a
            # durable copy. Copy bytes directly, without materializing a tensor.
            shutil.copyfile(self.path, path)
        return replace(self, path=str(path))

    def load(self, *, pin_memory: bool = False) -> torch.Tensor:
        path = Path(self.path)
        tensor = load_file(path, device="cpu").get("tensor")
        if tensor is None:
            raise OSError(f"Disk tensor file {path} does not contain the 'tensor' entry.")

        actual_nbytes = tensor.numel() * tensor.element_size()
        if (
            tuple(tensor.shape) != tuple(self.shape)
            or tensor.dtype != self.torch_dtype
            or actual_nbytes != self.nbytes
        ):
            raise OSError(
                f"Disk tensor metadata mismatch for {path}: file has "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, nbytes={actual_nbytes}; "
                f"reference expects shape={self.shape}, dtype={self.dtype}, nbytes={self.nbytes}."
            )

        if pin_memory and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor
