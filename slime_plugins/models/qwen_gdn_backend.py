import torch


def _parse_version(version):
    version = version.split("+", 1)[0]
    parts = version.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    return major, minor


def _validate_flashqla_runtime():
    if _parse_version(torch.__version__) < (2, 8):
        raise RuntimeError(f"FlashQLA backend requires PyTorch 2.8 or newer, got PyTorch {torch.__version__}.")

    if not torch.cuda.is_available():
        raise RuntimeError("FlashQLA backend requires CUDA and an NVIDIA SM90 GPU.")

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (9, 0):
        raise RuntimeError(f"FlashQLA backend requires NVIDIA SM90 or newer, got sm{major}{minor}.")

    cuda_version = torch.version.cuda
    if cuda_version is not None and _parse_version(cuda_version) < (12, 8):
        raise RuntimeError(f"FlashQLA backend requires CUDA 12.8 or newer, got CUDA {cuda_version}.")


def get_chunk_gated_delta_rule(backend: str):
    if backend == "fla":
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError("Qwen GDN backend 'fla' requires flash-linear-attention.") from exc
        return chunk_gated_delta_rule

    if backend == "flashqla":
        try:
            from flash_qla import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError(
                "Qwen GDN backend 'flashqla' requires FlashQLA. " "Install it from https://github.com/QwenLM/FlashQLA."
            ) from exc
        _validate_flashqla_runtime()
        return chunk_gated_delta_rule

    raise ValueError(f"Unsupported Qwen GDN backend: {backend}")
