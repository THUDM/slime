from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Sample:
    """The sample generated"""

    index: Optional[int] = None
    prompt: Optional[str] = None
    label: Optional[str] = None
    response: Optional[str] = None
    tokens: Optional[list[int]] = None
    response_length: Optional[int] = None
    truncated: Optional[bool] = None
    reward: Optional[float] = None
    loss_mask: Optional[list[int]] = None
    metadata: Optional[dict] = None
    version: int = 0
    aborted: bool = False


@dataclass
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int
