from __future__ import annotations

import errno
import math
import os
import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def retain_debug_tensor_refs(dump_data: dict, dump_path: Path) -> None:
    """Give a debug dump its own file lifetime, independent of rollout cleanup."""
    directory = dump_path.parent / (dump_path.stem + "_r3_spill") / uuid.uuid4().hex
    retained = {}
    for sample in dump_data.get("samples", []):
        for key in ("rollout_routed_experts", "rollout_topk_token_ids", "rollout_topk_log_probs"):
            ref = sample.get(key)
            if isinstance(ref, DiskTensorRef):
                if ref.path not in retained:
                    retained[ref.path] = ref.link(directory / f"{len(retained)}_{Path(ref.path).name}")
                sample[key] = retained[ref.path]


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

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows: slice) -> torch.Tensor:
        """Read only selected rows, so CP ranks never materialize a full sample."""
        if not isinstance(rows, slice) or rows.step not in (None, 1):
            raise TypeError("Disk tensor reads require a contiguous row slice.")
        start, stop, _ = rows.indices(len(self))
        if stop <= start:
            return torch.empty((0, *self.shape[1:]), dtype=self.torch_dtype)
        dtype_codes = {
            "uint8": "U8",
            "int8": "I8",
            "int16": "I16",
            "int32": "I32",
            "int64": "I64",
            "float16": "F16",
            "bfloat16": "BF16",
            "float32": "F32",
            "float64": "F64",
            "bool": "BOOL",
        }
        with safe_open(self.path, framework="pt", device="cpu") as file:
            value = file.get_slice("tensor")
            if (
                tuple(value.get_shape()) != tuple(self.shape)
                or value.get_dtype() != dtype_codes.get(self.dtype)
                or math.prod(self.shape) * torch.empty((), dtype=self.torch_dtype).element_size() != self.nbytes
            ):
                raise OSError(f"Disk tensor metadata mismatch for {self.path}")
            return value[start : max(start, stop)]

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
