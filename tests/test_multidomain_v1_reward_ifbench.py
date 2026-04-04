from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import types
from enum import Enum

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_reward_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "examples" / "multidomain_v1" / "reward_multidomain_v1.py"
    )
    module = types.ModuleType("reward_multidomain_v1_test_module")
    module.__file__ = str(module_path)
    source = module_path.read_text(encoding="utf-8")
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module


def _install_fake_ifbench_verifier():
    fake_module = types.ModuleType("evaluation_lib")
    fake_registry_module = types.ModuleType("instructions_registry")
    fake_registry_module.INSTRUCTION_DICT = {"count:keywords_multiple": object()}

    class InputExample:
        def __init__(self, key, instruction_id_list, prompt, kwargs) -> None:
            self.key = key
            self.instruction_id_list = instruction_id_list
            self.prompt = prompt
            self.kwargs = kwargs

    class Result:
        def __init__(self, follow_instruction_list):
            self.follow_instruction_list = list(follow_instruction_list)
            self.follow_all_instructions = all(self.follow_instruction_list)

    def test_instruction_following_loose(inp, prompt_to_response):
        assert inp.instruction_id_list == ["count:keywords_multiple"]
        assert inp.kwargs == [{"keyword1": "kaleidoscope", "keyword2": "nebula"}]
        assert inp.prompt in prompt_to_response
        return Result([True])

    fake_module.InputExample = InputExample
    fake_module.test_instruction_following_loose = test_instruction_following_loose
    fake_module.test_instruction_following_strict = test_instruction_following_loose

    original_module = sys.modules.get("evaluation_lib")
    original_registry_module = sys.modules.get("instructions_registry")
    sys.modules["evaluation_lib"] = fake_module
    sys.modules["instructions_registry"] = fake_registry_module
    return original_module, original_registry_module


def test_ifbench_reward_uses_ifbench_verifier_for_new_instruction_ids():
    class _Status(Enum):
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    class _Sample:
        Status = _Status

        def __init__(self, prompt, response, status, metadata):
            self.prompt = prompt
            self.response = response
            self.status = status
            self.metadata = metadata

    original_module, original_registry_module = _install_fake_ifbench_verifier()
    try:
        reward_module = _load_reward_module()
        reward_module._IFBENCH_REGISTRY = {"count:keywords_multiple": object()}
        sample = _Sample(
            prompt=[{"role": "system", "content": "Do task"}],
            response="kaleidoscope nebula",
            status=_Sample.Status.COMPLETED,
            metadata={
                "dataset_name": "ifbench_test",
                "reward_type": "instruction_following_soft",
                "prompt_text": "Do task",
                "instruction_id_list": ["count:keywords_multiple"],
                "kwargs": [{"keyword1": "kaleidoscope", "keyword2": "nebula", "N": None}],
            },
        )
        score = asyncio.run(reward_module.reward_func(None, sample))
    finally:
        if original_module is None:
            sys.modules.pop("evaluation_lib", None)
        else:
            sys.modules["evaluation_lib"] = original_module
        if original_registry_module is None:
            sys.modules.pop("instructions_registry", None)
        else:
            sys.modules["instructions_registry"] = original_registry_module

    assert score == 1.0


def test_ifbench_reward_fails_fast_without_official_verifier():
    class _Status(Enum):
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    class _Sample:
        Status = _Status

        def __init__(self, prompt, response, status, metadata):
            self.prompt = prompt
            self.response = response
            self.status = status
            self.metadata = metadata

    original_module = sys.modules.pop("evaluation_lib", None)
    original_registry_module = sys.modules.pop("instructions_registry", None)
    try:
        reward_module = _load_reward_module()
        reward_module._IFBENCH_REGISTRY = {}
        sample = _Sample(
            prompt=[{"role": "system", "content": "Do task"}],
            response="anything",
            status=_Sample.Status.COMPLETED,
            metadata={
                "dataset_name": "ifbench_test",
                "reward_type": "instruction_following_soft",
                "prompt_text": "Do task",
                "instruction_id_list": ["count:keywords_multiple"],
                "kwargs": [{"keyword1": "kaleidoscope"}],
            },
        )
        try:
            asyncio.run(reward_module.reward_func(None, sample))
            raised = None
        except RuntimeError as exc:
            raised = exc
    finally:
        if original_module is not None:
            sys.modules["evaluation_lib"] = original_module
        if original_registry_module is not None:
            sys.modules["instructions_registry"] = original_registry_module

    assert raised is not None
    assert "evaluation_lib" in str(raised)
