"""Patch sgl-kernel CMakeLists.txt to add a SGL_KERNEL_GB10_ONLY build option.

On GB10 (DGX Spark, sm_121a aarch64 with unified 128 GB memory), the stock
sgl-kernel CMake emits gencodes for sm_90a + sm_100a + sm_103a + sm_110a +
sm_120a + sm_121a. Cutlass FP8 gemm template instantiation per extra arch uses
10-15 GB RAM, and 12 parallel nvcc jobs OOM-kill cicc.

This patch wraps the non-GB10 gencode blocks in `if (NOT SGL_KERNEL_GB10_ONLY)`
and adds a GB10-only branch that emits just sm_120a + sm_121a. Default OFF
preserves upstream behavior.
"""

import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/src/sglang/sgl-kernel/CMakeLists.txt")
s = p.read_text()

marker = 'option(SGL_KERNEL_ENABLE_SM100A           "Enable SM100A"           OFF)'
new_opt = (
    marker
    + "\n"
    + 'option(SGL_KERNEL_GB10_ONLY              "Build only for GB10 (sm_121a + sm_120a). Skips sm_90a/sm_100a/sm_103a/sm_110a and FA3." OFF)'
)
assert marker in s, "marker 1 (SM100A option line) not found"
s = s.replace(marker, new_opt, 1)

old = 'if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)'
assert old in s, "marker 2 (CUDA 12.8 block) not found"
s = s.replace(old, "if (NOT SGL_KERNEL_GB10_ONLY)\n\n" + old, 1)

old2 = 'if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_FP4)'
assert old2 in s, "marker 3 (FP4 block) not found"
insert_before = """endif()  # NOT SGL_KERNEL_GB10_ONLY

if (SGL_KERNEL_GB10_ONLY)
    # GB10 (DGX Spark) is sm_121a (consumer Blackwell, aarch64). It cannot run
    # Hopper (sm_90a) nor datacenter Blackwell (sm_100a) binaries. Compiling
    # them wastes ~6x build time and ~10-15 GB RAM per TU (cutlass templates),
    # which OOM-kills cicc on 128 GB Spark hosts.
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_120a,code=sm_120a"
        "-gencode=arch=compute_121a,code=sm_121a"
        "--compress-mode=size"
    )
endif()

"""
s = s.replace(old2, insert_before + old2, 1)

p.write_text(s)
print(f"edits applied to {p}")
