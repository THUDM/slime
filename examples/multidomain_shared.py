from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

SLIME_ROOT = Path(__file__).resolve().parents[1]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

from examples.common.dataset_registry import (
    EVAL_DATASET_SPECS,
    default_train_datasets_for_domain,
    default_train_datasets_for_group,
    generic_eval_dataset_names,
    official_eval_dataset_names,
)
from examples.common.dataset_selection import (
    discover_canonical_train_sources,
    domain_signature,
    domain_signature_for_train_datasets,
    group_signature,
    group_signature_for_train_datasets,
    resolve_named_datasets,
    train_domains_for_datasets,
    train_groups_for_datasets,
)
from slime.rollout.rm_hub.ifbench import compute_ifbench_rule_scores
from slime.rollout.rm_hub.instruction_following_reward import normalize_instruction_ids, normalize_kwargs
from slime.rollout.rm_hub.multidomain_reward import compute_generic_reward as _compute_generic_reward
from slime.rollout.rm_hub.tool_call_reward import parse_predicted_calls as _base_parse_predicted_calls

logger = logging.getLogger(__name__)

GENERIC_EVAL_DATASETS: list[tuple[str, str, int]] = [
    (EVAL_DATASET_SPECS[name].name, EVAL_DATASET_SPECS[name].relpath, EVAL_DATASET_SPECS[name].n_samples_per_eval_prompt)
    for name in generic_eval_dataset_names()
]

OFFICIAL_EVAL_DATASETS: list[tuple[str, str]] = [
    (EVAL_DATASET_SPECS[name].name, EVAL_DATASET_SPECS[name].relpath)
    for name in official_eval_dataset_names()
]

_IFEVAL_BACKEND: dict[str, Any] | None = None


def _parse_predicted_calls(response: str, *, tools: list[dict[str, Any]], parser_type: str) -> list[dict[str, Any]]:
    return _base_parse_predicted_calls(response, tools=tools, parser_type=parser_type)


def _load_ifeval_backend() -> dict[str, Any]:
    global _IFEVAL_BACKEND
    if _IFEVAL_BACKEND is not None:
        return _IFEVAL_BACKEND
    attempts: list[str] = []

    try:
        eval_mod = importlib.import_module("instruction_following_eval.evaluation_lib")
        registry_mod = importlib.import_module("instruction_following_eval.instructions_registry")
        _IFEVAL_BACKEND = {
            "kind": "google",
            "InputExample": getattr(eval_mod, "InputExample"),
            "strict": getattr(eval_mod, "test_instruction_following_strict"),
            "loose": getattr(eval_mod, "test_instruction_following_loose"),
            "registry": getattr(registry_mod, "INSTRUCTION_DICT", None),
        }
        return _IFEVAL_BACKEND
    except Exception as exc:
        attempts.append(f"instruction_following_eval.evaluation_lib: {exc}")

    try:
        eval_mod = importlib.import_module("evaluation_lib")
        _IFEVAL_BACKEND = {
            "kind": "google",
            "InputExample": getattr(eval_mod, "InputExample"),
            "strict": getattr(eval_mod, "test_instruction_following_strict"),
            "loose": getattr(eval_mod, "test_instruction_following_loose"),
            "registry": None,
        }
        return _IFEVAL_BACKEND
    except Exception as exc:
        attempts.append(f"evaluation_lib: {exc}")

    try:
        ifeval_mod = importlib.import_module("ifeval")
        _IFEVAL_BACKEND = {
            "kind": "ifeval",
            "Evaluator": getattr(ifeval_mod, "Evaluator"),
            "InputExample": getattr(ifeval_mod, "InputExample"),
            "instruction_registry": getattr(ifeval_mod, "instruction_registry"),
        }
        return _IFEVAL_BACKEND
    except Exception as exc:
        attempts.append(f"ifeval: {exc}")

    raise RuntimeError(
        "Official IFEval backend is required. Supported imports: "
        "instruction_following_eval.evaluation_lib, evaluation_lib, or ifeval. "
        f"Tried: {'; '.join(attempts)}"
    )


def _ifbench_rule_scores(metadata: dict[str, Any], prompt_text: str, response: str, strict: bool) -> list[float] | None:
    payload = dict(metadata)
    payload["prompt_text"] = prompt_text
    return compute_ifbench_rule_scores(payload, response, strict=strict)


def _ifeval_rule_scores(metadata: dict[str, Any], prompt_text: str, response: str, strict: bool) -> list[float] | None:
    backend = _load_ifeval_backend()
    instruction_ids = normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        return []
    kwargs_list = normalize_kwargs(metadata.get("kwargs"), len(instruction_ids))
    raw_key = metadata.get("record_id", metadata.get("key", 0))
    try:
        key = int(raw_key)
    except (TypeError, ValueError):
        key = hash(str(raw_key)) % (10**9)

    if backend["kind"] == "google":
        try:
            input_example = backend["InputExample"](
                key=key,
                instruction_id_list=instruction_ids,
                prompt=prompt_text,
                kwargs=kwargs_list,
            )
            prompt_to_response = {prompt_text: response}
            verifier = backend["strict"] if strict else backend["loose"]
            result = verifier(input_example, prompt_to_response)
        except Exception:
            return None
        follow_list = getattr(result, "follow_instruction_list", None)
        if not isinstance(follow_list, list):
            return None
        return [1.0 if bool(item) else 0.0 for item in follow_list]

    try:
        evaluator = backend["Evaluator"](backend["instruction_registry"])
        input_example = backend["InputExample"](
            key=key,
            instruction_id_list=instruction_ids,
            prompt=prompt_text,
            kwargs=kwargs_list,
        )
        report, outputs = evaluator.evaluate([input_example], {prompt_text: response})
    except Exception:
        return None
    if not isinstance(outputs, list) or not outputs:
        return None
    follow_list = getattr(outputs[0], "follow_instruction_list", None)
    if not isinstance(follow_list, list):
        return None
    return [1.0 if bool(item) else 0.0 for item in follow_list]


def compute_generic_reward(metadata: dict[str, Any], prompt_text: str, response: str) -> float:
    return _compute_generic_reward(
        metadata,
        prompt_text,
        response,
        parse_predicted_calls_fn=_parse_predicted_calls,
        ifbench_rule_scores_fn=_ifbench_rule_scores,
        ifeval_rule_scores_fn=_ifeval_rule_scores,
    )


async def reward_func(args, sample, **kwargs) -> float:
    if sample.status == sample.Status.TRUNCATED:
        return 0.0
    metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
    if "tools" not in metadata:
        tools = getattr(sample, "tools", None)
        if isinstance(tools, list):
            metadata["tools"] = tools
    prompt = metadata.get("prompt_text") if metadata.get("prompt_text") not in (None, "") else (sample.prompt or "")
    if not isinstance(prompt, str):
        prompt = str(prompt)
    return compute_generic_reward(metadata, prompt, sample.response or "")
