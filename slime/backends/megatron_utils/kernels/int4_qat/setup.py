import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Get CUDA arch list
arch_list = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        arch_list.append(f"{major}.{minor}")
    arch_list = sorted(set(arch_list))

# Fallback to TORCH_CUDA_ARCH_LIST env var or default architectures when GPU is not available
if not arch_list:
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if env_arch:
        # Parse TORCH_CUDA_ARCH_LIST format: "7.0 7.5 8.0 8.6 9.0+PTX"
        arch_list = [a.strip().replace("+PTX", "") for a in env_arch.replace(";", " ").split() if a.strip()]
    else:
        # Default to common architectures (Volta, Turing, Ampere, Ada, Hopper)
        arch_list = ["8.0", "8.6", "8.9", "9.0"]

setup(
    name="fake_int4_quant_cuda",
    ext_modules=[
        CUDAExtension(
            name="fake_int4_quant_cuda",
            sources=["fake_int4_quant_cuda.cu"],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-Xcompiler",
                    "-fPIC",
                ]
                + [
                    f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}'
                    for arch in arch_list
                ]
                + ["-gencode=arch=compute_90a,code=sm_90a"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
