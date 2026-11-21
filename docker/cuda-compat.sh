#!/bin/bash
# Sourced automatically by the Docker entrypoint and conda activation hook.
# Older data-center drivers use the CUDA 13.0 user-space driver libraries;
# CUDA-13-capable hosts keep their own driver libraries.
_slime_enable_cuda_compat() {
    local mode="${SLIME_CUDA_COMPAT:-auto}"
    local compat_dir="${SLIME_CUDA_COMPAT_PATH:-/usr/local/cuda/compat}"
    local driver_version driver_major
    case "$mode" in
        0) return 0 ;;
        1) ;;
        auto)
            driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
            driver_major="${driver_version%%.*}"
            case "$driver_major" in
                535|550|565|570|575) ;;
                *) return 0 ;;
            esac
            ;;
        *) echo "SLIME_CUDA_COMPAT must be auto, 0 or 1; got: $mode" >&2; return 1 ;;
    esac
    if [ ! -r "$compat_dir/libcuda.so.1" ]; then
        echo "CUDA compatibility library not found: $compat_dir/libcuda.so.1" >&2
        return 1
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        ":$compat_dir:"*) ;;
        *) export LD_LIBRARY_PATH="$compat_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}
_slime_enable_cuda_compat
