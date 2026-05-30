"""HTTP adapters for agent rollouts."""

from slime.agent.adapters.anthropic import AnthropicAdapter
from slime.agent.adapters.common import BaseAdapter
from slime.agent.adapters.openai import OpenAIAdapter

__all__ = ["AnthropicAdapter", "BaseAdapter", "OpenAIAdapter"]
