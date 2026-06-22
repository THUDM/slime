"""Megatron teacher server utilities."""

from slime.backends.megatron_utils.server.arguments import (
    add_megatron_server_arguments,
    configure_megatron_server_args,
    validate_megatron_server_args,
)

__all__ = [
    "add_megatron_server_arguments",
    "configure_megatron_server_args",
    "validate_megatron_server_args",
]
