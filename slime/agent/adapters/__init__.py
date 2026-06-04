"""HTTP adapters for agent rollouts."""

from slime.agent.adapters.anthropic import AnthropicAdapter
from slime.agent.adapters.common import BaseAdapter

__all__ = ["AnthropicAdapter", "BaseAdapter"]
