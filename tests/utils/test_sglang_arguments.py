import argparse
import importlib
import sys
import types

import pytest

NUM_GPUS = 0


def _load_sglang_arguments(monkeypatch, request, server_args_cls):
    module_name = "slime.backends.sglang_utils.arguments"
    old_module = sys.modules.pop(module_name, None)

    def restore_module():
        sys.modules.pop(module_name, None)
        if old_module is not None:
            sys.modules[module_name] = old_module

    request.addfinalizer(restore_module)

    sglang_mod = types.ModuleType("sglang")
    srt_mod = types.ModuleType("sglang.srt")
    server_args_mod = types.ModuleType("sglang.srt.server_args")
    server_args_mod.ServerArgs = server_args_cls

    router_mod = types.ModuleType("sglang_router")
    launch_router_mod = types.ModuleType("sglang_router.launch_router")

    class RouterArgs:
        @staticmethod
        def add_cli_args(parser, use_router_prefix=True, exclude_host_port=True):
            return parser

    launch_router_mod.RouterArgs = RouterArgs

    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_mod)
    monkeypatch.setitem(sys.modules, "sglang_router", router_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router_mod)
    return importlib.import_module(module_name)


class ShortParallelServerArgs:
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--dp-size", dest="dp_size", type=int, default=1)
        parser.add_argument("--pp-size", dest="pp_size", type=int, default=1)
        parser.add_argument("--ep-size", dest="ep_size", type=int, default=1)
        parser.add_argument("--enable-dp-attention", action="store_true", default=False)


def test_validate_args_accepts_short_parallel_size_attrs(monkeypatch, request):
    sglang_arguments = _load_sglang_arguments(monkeypatch, request, ShortParallelServerArgs)

    args = argparse.Namespace(
        sglang_dp_size=2,
        sglang_pp_size=2,
        sglang_ep_size=4,
        sglang_enable_dp_attention=True,
        rollout_num_gpus_per_engine=4,
    )

    sglang_arguments.validate_args(args)

    assert args.sglang_data_parallel_size == 2
    assert args.sglang_pipeline_parallel_size == 2
    assert args.sglang_expert_parallel_size == 4
    assert args.sglang_tp_size == 2


def test_legacy_parallel_flags_work_with_short_sglang_dests(monkeypatch, request):
    sglang_arguments = _load_sglang_arguments(monkeypatch, request, ShortParallelServerArgs)
    parser = argparse.ArgumentParser()
    sglang_arguments.add_sglang_arguments(parser)

    args = parser.parse_args(
        [
            "--sglang-data-parallel-size",
            "2",
            "--sglang-pipeline-parallel-size",
            "2",
            "--sglang-expert-parallel-size",
            "4",
            "--sglang-enable-dp-attention",
        ]
    )
    args.rollout_num_gpus_per_engine = 4

    sglang_arguments.validate_args(args)

    assert args.sglang_dp_size == 2
    assert args.sglang_pp_size == 2
    assert args.sglang_ep_size == 4
    assert args.sglang_tp_size == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
