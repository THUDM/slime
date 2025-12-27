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
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
