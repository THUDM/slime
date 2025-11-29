import logging
import random
from argparse import Namespace
from copy import deepcopy
from typing import Any

from slime.rollout.sglang_rollout import generate as _generate_base
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


async def generate_with_random_osl(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    # TODO: make it configurable after we have an enhanced arg parser
    min_osl = 32 * 1024
    max_osl = 64 * 1024

    modified_sampling_params = deepcopy(sampling_params)
    modified_sampling_params["ignore_eos"] = True
    modified_sampling_params["max_new_tokens"] = random.randrange(min_osl, max_osl)

    ans = await _generate_base(args, sample, modified_sampling_params)

    logger.info(f"generate_with_random_osl {ans.response_length=}")
    return ans
