"""
SimpleTIR integration for slime examples.

This package provides utilities for running multi-turn, tool-integrated
rollouts and reward computation following the SimpleTIR recipe.
"""

__all__ = ["custom_generate", "async_reward"]


def __getattr__(name):
    if name == "custom_generate":
        from .generate import custom_generate

        return custom_generate
    if name == "async_reward":
        from .reward import async_reward

        return async_reward
    raise AttributeError(f"module {__name__} has no attribute {name!r}")
