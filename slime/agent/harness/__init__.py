"""Harness package facade.

External callers do ``from slime.agent.harness import CLAUDE_CODE``; the split
into one file per harness is internal.
"""

from __future__ import annotations

from .claude_code import CLAUDE_CODE, ClaudeCodeHarness
from .codex import CODEX, CodexHarness
from .common import BaseHarness, HarnessContext

__all__ = [
    "BaseHarness",
    "HarnessContext",
    "CLAUDE_CODE",
    "ClaudeCodeHarness",
    "CODEX",
    "CodexHarness",
]
