"""Logit-free linear cross entropy kernels vendored from verl.

Source: verl commit 4905d0cf4ebc7297231e15efa4cf837163efca45.
The source files retain their Apache-2.0 notices.
"""

from .linear_cross_entropy import linear_cross_entropy

__all__ = ["linear_cross_entropy"]
