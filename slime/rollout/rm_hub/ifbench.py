# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The follwoing code is adapted from the official IFBench code:
# https://github.com/allenai/IFBench/blob/main/evaluation_lib.py

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from .ifbench_utils import instructions_registry

logger = logging.getLogger(__name__)


JsonDict = Dict[str, Any]
KwargsDict = Dict[str, Optional[Union[str, int, float]]]


@dataclasses.dataclass
class InputExample:
    """Subset of the official InputExample schema needed for evaluation."""

    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[KwargsDict]


@dataclasses.dataclass
class OutputExample:
    """Official output structure for readability and parity."""

    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> List[str]:
    """Ensure instruction identifiers are clean strings."""

    normalized: List[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _coerce_kwargs_list(
    raw_kwargs: Any,
    num_instructions: int,
) -> List[KwargsDict]:
    """Convert stored kwargs into the list structure expected by IFBench."""

    if isinstance(raw_kwargs, list):
        processed: List[KwargsDict] = []
        for entry in raw_kwargs:
            if isinstance(entry, dict):
                processed.append(dict(entry))
            else:
                processed.append({})
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    # Remove explicit None values to match official preprocessing.
    sanitized: List[KwargsDict] = []
    for entry in processed:
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    return sanitized


def _build_input_example(metadata: JsonDict) -> Optional[InputExample]:
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list") or [])
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return None

    prompt_text = metadata.get("prompt_text")
    if prompt_text is None:
        prompt_text = ""
    else:
        prompt_text = str(prompt_text)

    raw_kwargs = metadata.get("kwargs")
    kwargs_list = _coerce_kwargs_list(raw_kwargs, len(instruction_ids))

    return InputExample(
        key=int(metadata.get("record_id") or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


def test_instruction_following_strict(inp: InputExample, response: str) -> OutputExample:
    """Official strict evaluation copied from evaluation_lib.py."""

    response = response or ""
    instruction_list = inp.instruction_id_list
    is_following_list: List[bool] = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            logger.warning("Unknown instruction id '%s'; marking as failed.", instruction_id)
            is_following_list.append(False)
            continue

        instruction = instruction_cls(instruction_id)
        kwargs = inp.kwargs[index] if index < len(inp.kwargs) else {}

        try:
            instruction.build_description(**kwargs)
        except Exception as exc:  # pragma: no cover - parity with official logic
            logger.debug("build_description failed for %s with kwargs %s: %s", instruction_id, kwargs, exc)
            instruction.build_description()

        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def compute_ifbench_reward(response: str, label: Any, metadata: Optional[JsonDict] = None) -> float:
    """Score a model response using the official IFBench rules."""

    if metadata is None:
        logger.debug("No metadata provided for IFBench scoring.")
        return 0.0

    if response is None:
        return 0.0

    inp = _build_input_example(metadata)
    if inp is None:
        return 0.0

    output = test_instruction_following_strict(inp, str(response))
    return 1.0 if output.follow_all_instructions else 0.0
