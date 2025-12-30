from dataclasses import dataclass


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None = None
