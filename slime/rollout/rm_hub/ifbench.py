from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_PARENT = _WORKSPACE_ROOT.parent
_LOCAL_IFBENCH_REQUIREMENTS = _WORKSPACE_ROOT / "examples" / "eval_multi_task" / "requirements_ifbench.txt"


def _ensure_ifbench_repo() -> Path:
    """Clone IFBench repo if needed and ensure it is available on sys.path."""

    repo_path = _WORKSPACE_PARENT / "IFBench"

    if not repo_path.exists():
        clone_cmd = ["git", "clone", "https://github.com/allenai/IFBench.git", str(repo_path)]
        try:
            subprocess.run(clone_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            raise ImportError(
                "Unable to automatically clone IFBench. Please clone "
                "https://github.com/allenai/IFBench.git into the repo root."
            ) from exc

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath is None:
        os.environ["PYTHONPATH"] = repo_str
    elif repo_str not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_str, current_pythonpath])

    return repo_path


def _ensure_ifbench_dependencies(repo_path: Path) -> None:
    """Install IFBench requirements the first time the module is imported."""

    requirements_file = _LOCAL_IFBENCH_REQUIREMENTS

    if not requirements_file.exists():
        logger.debug("Local IFBench requirements file not found at %s; skipping install.", requirements_file)
        return

    sentinel = repo_path / ".deps_installed"
    if sentinel.exists():
        return

    install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
    try:
        subprocess.run(install_cmd, check=True)
    except Exception as exc:
        logger.warning("Failed to install IFBench dependencies automatically: %s", exc)
    else:
        sentinel.write_text("installed\n")


def _load_evaluation_lib():
    repo_path = _ensure_ifbench_repo()
    try:
        return importlib.import_module("evaluation_lib")
    except ImportError:
        _ensure_ifbench_dependencies(repo_path)
        return importlib.import_module("evaluation_lib")


evaluation_lib = _load_evaluation_lib()
InputExample = evaluation_lib.InputExample


JsonDict = Dict[str, Any]
KwargsDict = Dict[str, Optional[Union[str, int, float]]]


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

    prompt_to_response = {inp.prompt: str(response or "")}
    output = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    return 1.0 if output.follow_all_instructions else 0.0
