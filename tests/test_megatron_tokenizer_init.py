"""Tokenizer cache transactions must be safe across training ranks."""

import ast
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import pytest
from filelock import FileLock

NUM_GPUS = 0


def load_init(cache_dir, build_tokenizer):
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/initialize.py"
    function = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "init"
    )
    namespace = {
        "set_args": lambda args: None,
        "_initialize_distributed": lambda args: None,
        "_set_random_seed": lambda *args: None,
        "_build_tokenizer": build_tokenizer,
        "init_num_microbatches_calculator": lambda *args: None,
        "np": types.SimpleNamespace(__version__="1.26.4"),
        "nullcontext": nullcontext,
        "Path": Path,
        "FileLock": FileLock,
        "HF_MODULES_CACHE": str(cache_dir),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace
    )
    return namespace["init"]


def args(tokenizer_type="HuggingFaceTokenizer"):
    return types.SimpleNamespace(
        enable_experimental=False,
        rank=1,
        seed=1234,
        data_parallel_random_init=False,
        te_rng_tracker=False,
        inference_rng_tracker=False,
        tokenizer_type=tokenizer_type,
        rampup_batch_size=None,
        global_batch_size=8,
        micro_batch_size=1,
        data_parallel_size=1,
        decrease_batch_size_if_needed=False,
        deterministic_mode=False,
        tp_comm_overlap=False,
    )


def test_ranks_cannot_import_partially_written_tokenizer_code(tmp_path):
    cache_dir = tmp_path / "modules"
    source = cache_dir / "tokenization_moonshot.py"
    copying, finish_copy, reader_started = threading.Event(), threading.Event(), threading.Event()
    loaded = []

    def build_tokenizer(_args):
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            with source.open("w") as output:
                copying.set()
                assert finish_copy.wait(5)
                output.write("class TikTokenTokenizer: pass\n")
        module = {}
        exec(source.read_text(), module)
        loaded.append(module["TikTokenTokenizer"]())

    init = load_init(cache_dir, build_tokenizer)

    def reader():
        reader_started.set()
        init(args())

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(init, args())
        try:
            assert copying.wait(5)
            concurrent_reader = executor.submit(reader)
            assert reader_started.wait(5)
            # The reader must wait for the copy/import transaction to finish.
            with pytest.raises(TimeoutError):
                concurrent_reader.result(timeout=0.2)
        finally:
            finish_copy.set()
        writer.result(timeout=5)
        concurrent_reader.result(timeout=5)
    assert len(loaded) == 2


@pytest.mark.parametrize("tokenizer_type", ["HuggingFaceTokenizer", "GPT2BPETokenizer"])
def test_tokenizer_build_failure_does_not_prevent_retry(tmp_path, tokenizer_type):
    calls = []

    def build_tokenizer(_args):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError("invalid tokenizer config")

    init = load_init(tmp_path / "modules", build_tokenizer)
    with pytest.raises(ValueError, match="invalid tokenizer config"):
        init(args(tokenizer_type))
    init(args(tokenizer_type))
    assert len(calls) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
