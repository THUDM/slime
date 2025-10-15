from dataclasses import dataclass
from typing import Optional


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: Optional[str] = None
