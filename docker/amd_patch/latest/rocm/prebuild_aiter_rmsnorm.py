#!/usr/bin/env python3
"""Build only AITER's module_rmsnorm_quant JIT extension.

Import ``jit.core`` through the source-tree path, matching AITER's setup.py.
Importing ``aiter.jit.core`` first executes ``aiter/__init__.py``, whose eager
attention imports query Triton's active GPU target. Standard Docker builds do
not expose a GPU driver, so avoid that runtime probe here.
"""

import os
import sys


aiter_root = os.getcwd()
os.environ["AITER_META_DIR"] = aiter_root
sys.path.insert(0, os.path.join(aiter_root, "aiter"))
from jit import core  # noqa: E402


MODULE = "module_rmsnorm_quant"
args = core.get_args_of_build(MODULE)

# Match the flags added by AITER's PREBUILD_KERNELS=1 setup.py path.
flags_extra_cc = [*args["flags_extra_cc"], "-DPREBUILD_KERNELS=1"]
flags_extra_hip = [*args["flags_extra_hip"], "-DPREBUILD_KERNELS=1"]

core.build_module(
    md_name=MODULE,
    srcs=args["srcs"],
    flags_extra_cc=flags_extra_cc,
    flags_extra_hip=flags_extra_hip,
    blob_gen_cmd=args["blob_gen_cmd"],
    extra_include=args["extra_include"],
    extra_ldflags=None,
    verbose=False,
    is_python_module=True,
    is_standalone=False,
    torch_exclude=False,
    third_party=args["third_party"],
)
