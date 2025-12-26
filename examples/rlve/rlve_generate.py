"""Custom generate function for RLVE - on-the-fly prompt generation from verifiable environments."""

import logging
from argparse import Namespace
from typing import Any

from slime.rollout.sglang_rollout import generate as _generate_base
from slime.utils.types import Sample

from .rlve_prompt_provider import get_provider

logger = logging.getLogger(__name__)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate prompt from RLVE environment, then call LLM for response."""
    provider = get_provider()
    prompt_bundle = provider.get_prompt()

    sample.prompt = prompt_bundle["prompt"]
    sample.metadata = prompt_bundle["metadata"]

    logger.debug(
        f"[RLVE] sample {sample.index} env={prompt_bundle['metadata']['env_id']} "
        f"prompt_len={len(sample.prompt)}"
    )

    sample = await _generate_base(args, sample, sampling_params)

    return sample
