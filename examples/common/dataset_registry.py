from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalDatasetSpec:
    name: str
    relpath: str
    n_samples_per_eval_prompt: int = 1
    runtime_mode: str = "pool"
    official: bool = False

    @property
    def domain(self) -> str:
        return self.relpath.split("/", 1)[0]


TRAIN_DATASET_SOURCE_MAP: dict[str, tuple[str, ...]] = {
    "apibench": (
        "tool/train/apibench_huggingface_train.jsonl",
        "tool/train/apibench_tensorflow_train.jsonl",
        "tool/train/apibench_torchhub_train.jsonl",
    ),
    "xlam_function_calling_60k": ("tool/train/xlam_function_calling_60k_xlam-function-calling-60k.jsonl",),
    "agent": (
        "tool/train/agent_function_calling_open_dataset_deepnlp_agent_function_call_202510.jsonl",
        "tool/train/agent_function_calling_open_dataset_deepnlp_agent_function_call_202601.jsonl",
    ),
    "jsonschemabench": ("structured/train/jsonschemabench_train-00000-of-00001.jsonl",),
    "nemotron_structured_outputs": ("structured/train/nemotron_structured_outputs_structured_outputs_251027_nano_v3_sdg_json_train.jsonl",),
    "nemotron_knowledge_mcqa": (
        "stem/train/nemotron_knowledge_mcqa_data_train-00000-of-00004.jsonl",
        "stem/train/nemotron_knowledge_mcqa_data_train-00001-of-00004.jsonl",
        "stem/train/nemotron_knowledge_mcqa_data_train-00002-of-00004.jsonl",
        "stem/train/nemotron_knowledge_mcqa_data_train-00003-of-00004.jsonl",
    ),
    "medmcqa": ("stem/train/medmcqa_data_train-00000-of-00001.jsonl",),
    "apps": ("code/apps.jsonl",),
    "code_contests": ("code/code_contests.jsonl",),
    "codeforces": ("code/codeforces.jsonl",),
    "taco": ("code/taco.jsonl",),
    "bigcodebench": ("code/bigcodebench.jsonl",),
    "deepmath_103k": ("math/deepmath-103k.jsonl",),
    "dapo_17k": ("math/dapo-17k.jsonl",),
    "bigmath_rl_verified": ("math/bigmath-rl-verified.jsonl",),
}

TRAIN_DATASET_GROUP_MAP: dict[str, str] = {
    "apibench": "tool_call",
    "xlam_function_calling_60k": "tool_call",
    "agent": "tool_call",
    "jsonschemabench": "structured",
    "nemotron_structured_outputs": "structured",
    "nemotron_knowledge_mcqa": "stem",
    "medmcqa": "stem",
    "apps": "code",
    "code_contests": "code",
    "codeforces": "code",
    "taco": "code",
    "bigcodebench": "code",
    "deepmath_103k": "math",
    "dapo_17k": "math",
    "bigmath_rl_verified": "math",
}

TRAIN_DATASET_DOMAIN_MAP: dict[str, str] = {
    dataset_name: relpaths[0].split("/", 1)[0] for dataset_name, relpaths in TRAIN_DATASET_SOURCE_MAP.items()
}

DEFAULT_TRAIN_DATASETS_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    "tool": ("apibench", "xlam_function_calling_60k", "agent"),
    "structured": ("jsonschemabench", "nemotron_structured_outputs"),
    "stem": ("medmcqa", "nemotron_knowledge_mcqa"),
    "code": ("apps", "code_contests", "codeforces", "taco"),
    "math": ("deepmath_103k", "dapo_17k", "bigmath_rl_verified"),
}

DEFAULT_TRAIN_DATASETS_BY_GROUP: dict[str, tuple[str, ...]] = {
    "tool_call": DEFAULT_TRAIN_DATASETS_BY_DOMAIN["tool"],
    "structured": DEFAULT_TRAIN_DATASETS_BY_DOMAIN["structured"],
    "stem": DEFAULT_TRAIN_DATASETS_BY_DOMAIN["stem"],
    "code": DEFAULT_TRAIN_DATASETS_BY_DOMAIN["code"],
    "math": DEFAULT_TRAIN_DATASETS_BY_DOMAIN["math"],
}

PROFILE_DEFAULT_TRAIN_DATASETS: dict[str, tuple[str, ...]] = {
    "mdv2": (
        "apibench",
        "xlam_function_calling_60k",
        "agent",
        "jsonschemabench",
        "nemotron_structured_outputs",
        "medmcqa",
        "nemotron_knowledge_mcqa",
    ),
    "mopd": (
        "apibench",
        "xlam_function_calling_60k",
        "agent",
        "jsonschemabench",
        "nemotron_structured_outputs",
        "medmcqa",
        "nemotron_knowledge_mcqa",
        "apps",
        "code_contests",
        "codeforces",
        "taco",
        "deepmath_103k",
        "dapo_17k",
        "bigmath_rl_verified",
    ),
}

_EVAL_DATASETS = (
    EvalDatasetSpec("aime24", "math/aime24.jsonl", n_samples_per_eval_prompt=32, runtime_mode="math"),
    EvalDatasetSpec("aime25", "math/aime25.jsonl", n_samples_per_eval_prompt=32, runtime_mode="math"),
    EvalDatasetSpec("amc23", "math/amc23.jsonl", n_samples_per_eval_prompt=32, runtime_mode="math"),
    EvalDatasetSpec("math500", "math/math500.jsonl", runtime_mode="math"),
    EvalDatasetSpec("olympiadmath", "math/olympiadmath.jsonl", runtime_mode="math"),
    EvalDatasetSpec("minerva", "math/minerva.jsonl", runtime_mode="math"),
    EvalDatasetSpec("livecodebench", "code/livecodebench.jsonl", runtime_mode="code"),
    EvalDatasetSpec("humanevalplus", "code/humanevalplus.jsonl", runtime_mode="code"),
    EvalDatasetSpec("mbppplus", "code/mbppplus.jsonl", runtime_mode="code"),
    EvalDatasetSpec("mmlu_pro", "stem/eval/mmlu_pro_test-00000-of-00001.jsonl"),
    EvalDatasetSpec("gpqa", "stem/eval/gpqa_gpqa_main.jsonl"),
    EvalDatasetSpec("jsonschemabench", "structured/eval/jsonschemabench_test-00000-of-00001.jsonl"),
    EvalDatasetSpec("ifeval", "structured/eval/ifeval_ifeval_input_data.jsonl"),
    EvalDatasetSpec("ifbench_test", "structured/eval/ifbench_test_data_train-00000-of-00001.jsonl"),
    EvalDatasetSpec("bfcl_v3", "tool/eval/bfcl_v3_train-00000-of-00001.jsonl", official=True),
    EvalDatasetSpec("bfcl_v3_multi_turn_base", "tool/eval/bfcl_v3_multi_turn_base_train-00000-of-00001.jsonl", official=True),
)

EVAL_DATASET_SPECS: dict[str, EvalDatasetSpec] = {spec.name: spec for spec in _EVAL_DATASETS}

PROFILE_DEFAULT_EVAL_DATASETS: dict[str, tuple[str, ...]] = {
    "mdv2": (
        "ifeval",
        "jsonschemabench",
        "ifbench_test",
        "mmlu_pro",
        "gpqa",
    ),
    "mopd": (
        "aime24",
        "aime25",
        "amc23",
        "math500",
        "olympiadmath",
        "minerva",
        "livecodebench",
        "humanevalplus",
        "mbppplus",
        "mmlu_pro",
        "gpqa",
        "jsonschemabench",
        "ifeval",
        "ifbench_test",
    ),
}


def default_train_datasets_for_domain(domain: str) -> tuple[str, ...]:
    try:
        return DEFAULT_TRAIN_DATASETS_BY_DOMAIN[domain]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_TRAIN_DATASETS_BY_DOMAIN))
        raise ValueError(f"Unsupported train domain '{domain}'. Supported domains: {supported}") from exc


def default_train_datasets_for_group(group: str) -> tuple[str, ...]:
    try:
        return DEFAULT_TRAIN_DATASETS_BY_GROUP[group]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_TRAIN_DATASETS_BY_GROUP))
        raise ValueError(f"Unsupported train group '{group}'. Supported groups: {supported}") from exc


def default_train_datasets_for_profile(profile: str) -> tuple[str, ...]:
    try:
        return PROFILE_DEFAULT_TRAIN_DATASETS[profile]
    except KeyError as exc:
        supported = ", ".join(sorted(PROFILE_DEFAULT_TRAIN_DATASETS))
        raise ValueError(f"Unsupported profile '{profile}'. Supported profiles: {supported}") from exc


def default_eval_datasets_for_profile(profile: str) -> tuple[str, ...]:
    try:
        return PROFILE_DEFAULT_EVAL_DATASETS[profile]
    except KeyError as exc:
        supported = ", ".join(sorted(PROFILE_DEFAULT_EVAL_DATASETS))
        raise ValueError(f"Unsupported profile '{profile}'. Supported profiles: {supported}") from exc


def generic_eval_dataset_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in _EVAL_DATASETS if not spec.official)


def official_eval_dataset_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in _EVAL_DATASETS if spec.official)


def eval_dataset_spec(name: str) -> EvalDatasetSpec:
    try:
        return EVAL_DATASET_SPECS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(EVAL_DATASET_SPECS))
        raise ValueError(f"Unsupported eval dataset '{name}'. Supported datasets: {supported}") from exc
