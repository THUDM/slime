"""Select the sandbox used for both coding rollouts and fresh-image grading."""

import os

from slime.agent.sandbox import E2BSandbox, Sandbox


def create_sandbox(image: str) -> Sandbox:
    provider = os.environ.get("SWE_SANDBOX_PROVIDER", "sunabako")
    if provider == "e2b":
        return E2BSandbox(image)
    if provider == "sunabako":
        from .sunabako_sandbox import SunabakoSandbox

        return SunabakoSandbox(image)
    raise ValueError(f"Unknown SWE_SANDBOX_PROVIDER={provider!r}; expected e2b or sunabako")
