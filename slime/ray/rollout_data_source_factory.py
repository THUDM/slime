"""
Factory for selecting the appropriate buffer implementation.

Supports:
- in_process: Fast, embedded buffer (default for GRPO/PPO)
- http: HTTP-based buffer (for agent tasks and async generation)

Usage:
    from slime.ray.rollout_data_source_factory import create_rollout_data_source

    data_source = create_rollout_data_source(args)

"""

from slime.ray.rollout_data_source import RolloutDataSourceWithBuffer
from slime.ray.rollout_data_source_http import RolloutDataSourceWithHTTPBuffer


def create_rollout_data_source(args):
    """
    Factory function to create the appropriate rollout data source.

    Args:
        args: Training arguments with buffer configuration

    Returns:
        RolloutDataSource instance (with or without buffer)

    Configuration:
        --buffer_mode in_process  # Default, fast embedded buffer
        --buffer_mode http        # HTTP buffer for agent tasks
        --buffer_mode none        # No buffer (read-only)

    Examples:
        # Standard GRPO (in-process buffer)
        args.buffer_mode = 'in_process'
        data_source = create_rollout_data_source(args)

        # Agent tasks (HTTP buffer)
        args.buffer_mode = 'http'
        args.buffer_server_url = 'http://localhost:8889'
        args.buffer_task_type = 'grpo'  # or 'math', 'tool', etc.
        data_source = create_rollout_data_source(args)
    """
    buffer_mode = getattr(args, 'buffer_mode', 'in_process')

    if buffer_mode == 'http':
        print(f"[Buffer Factory] Using HTTP buffer mode")
        return RolloutDataSourceWithHTTPBuffer(args)
    elif buffer_mode == 'in_process':
        print(f"[Buffer Factory] Using in-process buffer mode")
        return RolloutDataSourceWithBuffer(args)
    elif buffer_mode == 'none':
        # No buffer, just the base class
        from slime.ray.rollout_data_source import RolloutDataSource
        print(f"[Buffer Factory] Using no buffer (read-only)")
        return RolloutDataSource(args)
    else:
        raise ValueError(
            f"Unknown buffer_mode: {buffer_mode}. "
            f"Valid options: 'in_process', 'http', 'none'"
        )
