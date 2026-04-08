from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import multiprocessing
import os
import pickle
import re
import sys
import textwrap
import traceback
import zlib
from collections.abc import Iterable
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 8.0
MAX_TIMEOUT_SECONDS = 20.0

_PRELUDE = """
import sys
sys.setrecursionlimit(6 * 10 ** 5)
"""

_TEST_CODE_HARNESS_RE = re.compile(
    r"(?P<prelude>.*)\n"
    r"for\s+i,\s*(?P<target>.+?)\s+in\s+enumerate\((?P<iter>.+?)\):\s*\n"
    r"(?P<body>(?:    .*\n?)*)\s*$",
    flags=re.DOTALL,
)
_CHECK_FUNCTION_HARNESS_RE = re.compile(
    r"(?P<prelude>.*)\n"
    r"def\s+check\((?P<candidate>\w+)\):\s*\n"
    r"(?P<setup>(?:    .*\n)*?)"
    r"    for\s+i,\s*(?P<target>.+?)\s+in\s+enumerate\((?P<iter>.+?)\):\s*\n"
    r"(?P<body>(?:        .*\n?)*)\s*$",
    flags=re.DOTALL,
)


def extract_python_code(completion: str) -> str:
    text = str(completion or "").strip()
    if not text:
        return ""

    python_fences = re.findall(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if python_fences:
        return python_fences[-1].strip()

    generic_fences = re.findall(r"```(?:py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if generic_fences:
        return generic_fences[-1].strip()

    return text


def normalize_output_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def compare_text_output(actual: Any, expected: Any) -> bool:
    actual_norm = normalize_output_text(actual)
    expected_norm = normalize_output_text(expected)
    if actual_norm == expected_norm:
        return True
    actual_tokens = actual_norm.split()
    expected_tokens = expected_norm.split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    if actual_tokens == expected_tokens:
        return True
    for left, right in zip(actual_tokens, expected_tokens, strict=False):
        try:
            if not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6):
                return False
        except Exception:
            return False
    return True


def expected_text_candidates(expected: Any) -> list[str]:
    if isinstance(expected, list):
        candidates: list[str] = []
        if all(not isinstance(item, (list, dict)) for item in expected):
            candidates.append("\n".join("" if item is None else str(item) for item in expected))
            for item in expected:
                candidates.append("" if item is None else str(item))
        return [candidate for candidate in candidates if candidate is not None]
    return ["" if expected is None else str(expected)]


def normalize_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, set):
        return sorted((normalize_value(item) for item in value), key=repr)
    if isinstance(value, dict):
        return {str(key): normalize_value(val) for key, val in value.items()}
    return value


def compare_values(actual: Any, expected: Any) -> bool:
    actual = normalize_value(actual)
    expected = normalize_value(expected)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(actual, str) and isinstance(expected, (int, float)):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
        except Exception:
            pass
    if isinstance(expected, str) and isinstance(actual, (int, float)):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
        except Exception:
            pass
    if isinstance(actual, str) and isinstance(expected, str):
        return compare_text_output(actual, expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(compare_values(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(compare_values(actual[k], expected[k]) for k in actual)
    return actual == expected


def resolve_timeout_seconds(metadata: dict[str, Any]) -> float:
    candidate = metadata.get("time_limit_seconds")
    if isinstance(candidate, (int, float)) and math.isfinite(candidate):
        return max(2.0, min(MAX_TIMEOUT_SECONDS, float(candidate) * 2.0 + 1.0))
    return DEFAULT_TIMEOUT_SECONDS


def parse_test_code_harness(test_code: str) -> dict[str, str] | None:
    text = str(test_code or "").strip()
    if not text:
        return None
    match = _TEST_CODE_HARNESS_RE.match(text)
    if match is not None:
        body = textwrap.dedent(match.group("body")).strip()
        if not body:
            return None
        return {
            "kind": "top_level_loop",
            "prelude": match.group("prelude").rstrip(),
            "setup_code": "",
            "loop_target": match.group("target").strip(),
            "loop_iter": match.group("iter").strip(),
            "loop_body": body,
            "candidate_param": "",
        }

    match = _CHECK_FUNCTION_HARNESS_RE.match(text)
    if match is None:
        return None
    body = textwrap.dedent(match.group("body")).strip()
    if not body:
        return None
    return {
        "kind": "check_function",
        "prelude": match.group("prelude").rstrip(),
        "setup_code": textwrap.dedent(match.group("setup")).strip(),
        "loop_target": match.group("target").strip(),
        "loop_iter": match.group("iter").strip(),
        "loop_body": body,
        "candidate_param": match.group("candidate").strip(),
    }


def decode_livecodebench_private_tests(payload_text: str, payload_format: str | None = None) -> list[dict[str, Any]]:
    if not payload_text:
        return []

    def _coerce_to_tests(restored: Any) -> list[dict[str, Any]]:
        if isinstance(restored, bytes):
            restored = restored.decode("utf-8")
        if isinstance(restored, str):
            restored = json.loads(restored)
        if isinstance(restored, list):
            return [item for item in restored if isinstance(item, dict)]
        return []

    if payload_format == "json":
        try:
            return _coerce_to_tests(json.loads(payload_text))
        except Exception:
            return []

    try:
        return _coerce_to_tests(json.loads(payload_text))
    except Exception:
        pass

    try:
        payload = zlib.decompress(base64.b64decode(payload_text))
        restored = pickle.loads(payload)
        return _coerce_to_tests(restored)
    except Exception:
        return []


def _tests_to_verifier(tests: Iterable[dict[str, Any]], fn_name: str | None = None) -> dict[str, Any]:
    inputs: list[Any] = []
    outputs: list[Any] = []
    for test in tests:
        if not isinstance(test, dict):
            continue
        if "input" not in test or "output" not in test:
            continue
        inputs.append(test.get("input"))
        outputs.append(test.get("output"))
    return {"inputs": inputs, "outputs": outputs, "fn_name": fn_name or None}


def build_verifier_from_metadata(metadata: dict[str, Any], *, include_private: bool = True) -> dict[str, Any]:
    verifier = metadata.get("verifier")
    if isinstance(verifier, dict):
        inputs = list(verifier.get("inputs") or [])
        outputs = list(verifier.get("outputs") or [])
        fn_name = verifier.get("fn_name")
        if include_private:
            private_tests = decode_livecodebench_private_tests(
                str(metadata.get("private_tests_encoded") or ""),
                payload_format=str(metadata.get("private_tests_format") or "").strip() or None,
            )
            private_verifier = _tests_to_verifier(private_tests, fn_name if isinstance(fn_name, str) else None)
            inputs.extend(private_verifier["inputs"])
            outputs.extend(private_verifier["outputs"])
        return {"inputs": inputs, "outputs": outputs, "fn_name": fn_name or None}

    tests = list(metadata.get("tests") or [])
    fn_name = str(metadata.get("fn_name") or "").strip() or None
    if include_private:
        tests.extend(
            decode_livecodebench_private_tests(
                str(metadata.get("private_tests_encoded") or ""),
                payload_format=str(metadata.get("private_tests_format") or "").strip() or None,
            )
        )
    return _tests_to_verifier(tests, fn_name=fn_name)


def slice_verifier(verifier: dict[str, Any], limit: int) -> dict[str, Any]:
    return {
        "inputs": list(verifier.get("inputs") or [])[:limit],
        "outputs": list(verifier.get("outputs") or [])[:limit],
        "fn_name": verifier.get("fn_name"),
    }


def _prepare_call_args(raw_input: Any) -> list[Any]:
    if isinstance(raw_input, list):
        return raw_input
    if isinstance(raw_input, tuple):
        return list(raw_input)
    return [raw_input]


def _resolve_callable(namespace: dict[str, Any], fn_name: str) -> Any:
    solution_cls = namespace.get("Solution")
    if solution_cls is not None and hasattr(solution_cls, fn_name):
        instance = solution_cls()
        method = getattr(instance, fn_name)
        if callable(method):
            return method

    fn = namespace.get(fn_name)
    if callable(fn):
        return fn

    raise NameError(f"Function {fn_name!r} was not defined by the candidate solution.")


def _run_call_case(code: str, fn_name: str, raw_input: Any, expected: Any) -> tuple[bool, dict[str, Any]]:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(_PRELUDE + "\n" + code, namespace, namespace)
    fn = _resolve_callable(namespace, fn_name)
    args = _prepare_call_args(raw_input)
    result = fn(*args)
    success = compare_values(result, expected)
    return success, {
        "mode": "call",
        "inputs": repr(raw_input),
        "expected": repr(expected),
        "output": repr(result),
        "fn_name": fn_name,
        "error_message": None if success else "Wrong Answer",
    }


def _run_stdin_case(code: str, raw_input: Any, expected: Any) -> tuple[bool, dict[str, Any]]:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    stdin_text = "" if raw_input is None else str(raw_input)
    fake_stdin = io.StringIO(stdin_text)
    fake_stdout = io.StringIO()

    with contextlib.redirect_stdout(fake_stdout), contextlib.redirect_stderr(io.StringIO()):
        original_stdin = sys.stdin
        original_stdout = sys.stdout
        sys.stdin = fake_stdin
        sys.stdout = fake_stdout
        try:
            exec(_PRELUDE + "\n" + code, namespace, namespace)
        finally:
            sys.stdin = original_stdin
            sys.stdout = original_stdout

    actual = fake_stdout.getvalue()
    success = any(compare_text_output(actual, candidate) for candidate in expected_text_candidates(expected))
    return success, {
        "mode": "stdin",
        "inputs": stdin_text,
        "expected": repr(expected),
        "output": actual,
        "error_message": None if success else "Wrong Answer",
    }


def run_test(verifier: dict[str, Any], generation: str) -> tuple[list[bool | int], dict[str, Any]]:
    inputs = list(verifier.get("inputs") or [])
    outputs = list(verifier.get("outputs") or [])
    fn_name = str(verifier.get("fn_name") or "").strip()
    if not inputs or len(inputs) != len(outputs):
        return [], {"error_message": "Invalid verifier payload"}

    results: list[bool | int] = []
    last_metadata: dict[str, Any] = {}
    for raw_input, expected in zip(inputs, outputs, strict=False):
        try:
            if fn_name:
                success, last_metadata = _run_call_case(generation, fn_name, raw_input, expected)
            else:
                success, last_metadata = _run_stdin_case(generation, raw_input, expected)
            results.append(bool(success))
            if not success:
                return results, last_metadata
        except Exception as exc:
            last_metadata = {
                "mode": "call" if fn_name else "stdin",
                "inputs": repr(raw_input),
                "expected": repr(expected),
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=10),
            }
            results.append(-1)
            return results, last_metadata

    return results, last_metadata


def _check_correctness_worker(
    verifier: dict[str, Any],
    generation: str,
    result_holder,
    metadata_holder,
) -> None:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            results, metadata = run_test(verifier, generation)
            result_holder.append(results)
            metadata_holder.append(metadata)
        except Exception:
            result_holder.append([-1 for _ in range(len(verifier.get("inputs") or []))])
            metadata_holder.append({"traceback": traceback.format_exc(limit=10)})


def check_correctness(
    verifier: dict[str, Any],
    generation: str,
    *,
    timeout: float,
) -> tuple[list[bool | int], list[dict[str, Any]]]:
    manager = multiprocessing.Manager()
    result_holder = manager.list()
    metadata_holder = manager.list()
    process = multiprocessing.Process(
        target=_check_correctness_worker,
        args=(verifier, generation, result_holder, metadata_holder),
    )
    process.start()
    process.join(timeout=timeout + 1.0)
    if process.is_alive():
        process.kill()
    if not result_holder:
        inputs = list(verifier.get("inputs") or [])
        return ([-1 for _ in inputs], [{"error_message": "Global timeout"}])
    return result_holder[0], list(metadata_holder)


def run_test_code_harness(
    harness: dict[str, str],
    generation: str,
    *,
    metadata: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[bool | int], dict[str, Any]]:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(_PRELUDE + "\n" + harness["prelude"] + "\n" + generation, namespace, namespace)

    if harness.get("kind") == "check_function":
        metadata = metadata or {}
        entry_point = str(metadata.get("entry_point") or metadata.get("fn_name") or "").strip()
        if not entry_point:
            return [], {"error_message": "Missing entry point for check(candidate) harness"}
        namespace[harness["candidate_param"]] = _resolve_callable(namespace, entry_point)
        setup_code = harness.get("setup_code", "").strip()
        if setup_code:
            exec(setup_code, namespace, namespace)

    iterator = eval(harness["loop_iter"], namespace, namespace)
    if not isinstance(iterator, Iterable):
        return [], {"error_message": "Harness iterator is not iterable"}

    results: list[bool | int] = []
    last_metadata: dict[str, Any] = {}
    for i, case_item in enumerate(iterator):
        if limit is not None and i >= limit:
            break
        try:
            namespace["i"] = i
            namespace["__case_item__"] = case_item
            exec(f"{harness['loop_target']} = __case_item__", namespace, namespace)
            exec(harness["loop_body"], namespace, namespace)
            results.append(True)
            last_metadata = {"mode": "test_code", "case_index": i, "error_message": None}
        except AssertionError:
            last_metadata = {"mode": "test_code", "case_index": i, "error_message": "Wrong Answer"}
            results.append(False)
            return results, last_metadata
        except Exception as exc:
            last_metadata = {
                "mode": "test_code",
                "case_index": i,
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=10),
            }
            results.append(-1)
            return results, last_metadata

    return results, last_metadata


def _check_test_code_harness_worker(
    harness: dict[str, str],
    generation: str,
    metadata: dict[str, Any] | None,
    limit: int | None,
    result_queue,
) -> None:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            results, details = run_test_code_harness(harness, generation, metadata=metadata, limit=limit)
            result_queue.put((results, [details]))
        except Exception:
            result_queue.put(([-1], [{"traceback": traceback.format_exc(limit=10)}]))


def check_test_code_harness_correctness(
    harness: dict[str, str],
    generation: str,
    *,
    timeout: float,
    metadata: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[bool | int], list[dict[str, Any]]]:
    result_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
    process = multiprocessing.Process(
        target=_check_test_code_harness_worker,
        args=(harness, generation, metadata, limit, result_queue),
    )
    process.start()
    process.join(timeout + 1.0)
    if process.is_alive():
        process.kill()
    if result_queue.empty():
        return ([-1], [{"error_message": "Global timeout"}])
    results, details = result_queue.get()
    return results, details


def compute_score(
    completion: str,
    metadata: dict[str, Any],
    *,
    continuous: bool = True,
    max_partial_cases: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    solution = extract_python_code(completion)
    if not solution:
        return 0.0, [{"error_message": "No code extracted"}]

    timeout = resolve_timeout_seconds(metadata)
    harness = parse_test_code_harness(str(metadata.get("test_code") or ""))
    if harness is not None:
        full_results, full_metadata = check_test_code_harness_correctness(
            harness,
            solution,
            timeout=timeout,
            metadata=metadata,
        )
        if full_results and all(item is True for item in full_results):
            return 1.0, full_metadata

        if not continuous:
            return 0.0, full_metadata

        partial_results, partial_metadata = check_test_code_harness_correctness(
            harness,
            solution,
            timeout=max(timeout, 10.0),
            metadata=metadata,
            limit=max_partial_cases,
        )
        if not partial_results:
            return 0.0, partial_metadata
        passed = sum(1 for item in partial_results if item is True)
        return passed / max(1, len(partial_results)), partial_metadata

    verifier = build_verifier_from_metadata(metadata, include_private=True)
    inputs = list(verifier.get("inputs") or [])
    outputs = list(verifier.get("outputs") or [])
    if not inputs or len(inputs) != len(outputs):
        return 0.0, [{"error_message": "Empty or invalid verifier"}]

    full_results, full_metadata = check_correctness(verifier, solution, timeout=timeout)
    if full_results and all(item is True for item in full_results):
        return 1.0, full_metadata

    if not continuous:
        return 0.0, full_metadata

    partial_verifier = slice_verifier(verifier, max_partial_cases)
    partial_results, partial_metadata = check_correctness(partial_verifier, solution, timeout=max(timeout, 10.0))
    if not partial_results:
        return 0.0, partial_metadata
    passed = sum(1 for item in partial_results if item is True)
    return passed / max(1, len(partial_results)), partial_metadata
