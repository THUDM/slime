from argparse import ArgumentParser, Namespace

import pytest

from slime.backends.sglang_utils.arguments import add_sglang_arguments, validate_args

NUM_GPUS = 0


def test_sglang_python_executable_is_a_launcher_option():
    parser = ArgumentParser()
    add_sglang_arguments(parser)
    assert parser.parse_args([]).sglang_python_executable is None
    executable = "/path with spaces/serve-python.sh"
    assert parser.parse_args(["--sglang-python-executable", executable]).sglang_python_executable == executable


def test_validate_args_canonicalizes_moe_data_parallel_size():
    args = Namespace(
        sglang_data_parallel_size=8,
        sglang_pipeline_parallel_size=7,
        sglang_expert_parallel_size=8,
        sglang_moe_data_parallel_size=1,
        rollout_num_gpus_per_engine=56,
        sglang_enable_dp_attention=True,
        sglang_router_ip=None,
        prefill_num_servers=None,
        rollout_external=False,
        sglang_config=None,
    )

    validate_args(args)

    assert args.sglang_tp_size == 8
    assert args.sglang_pp_size == 7
    assert args.sglang_dp_size == 8
    assert args.sglang_ep_size == 8
    assert args.sglang_moe_dp_size == 1


@pytest.mark.parametrize(
    "flags",
    [
        ["--sglang-dp-size", "--sglang-pp-size", "--sglang-ep-size", "--sglang-moe-dp-size"],
        [
            "--sglang-data-parallel-size",
            "--sglang-pipeline-parallel-size",
            "--sglang-expert-parallel-size",
            "--sglang-moe-data-parallel-size",
        ],
    ],
)
def test_parallel_cli_aliases_remain_compatible(flags):
    parser = ArgumentParser()
    add_sglang_arguments(parser)
    argv = [item for flag, value in zip(flags, (8, 7, 8, 1), strict=True) for item in (flag, str(value))]
    args = parser.parse_args([*argv, "--sglang-enable-dp-attention"])
    args.rollout_num_gpus_per_engine = 56
    validate_args(args)
    for current, legacy, expected in (
        ("dp_size", "data_parallel_size", 8),
        ("pp_size", "pipeline_parallel_size", 7),
        ("ep_size", "expert_parallel_size", 8),
        ("moe_dp_size", "moe_data_parallel_size", 1),
    ):
        assert getattr(args, f"sglang_{current}") == expected
        assert getattr(args, f"sglang_{legacy}") == expected


@pytest.mark.parametrize(
    "legacy,current,values,dest",
    [
        ("cuda-graph-max-bs", "cuda-graph-max-bs-decode", ["8"], "cuda_graph_max_bs_decode"),
        ("cuda-graph-bs", "cuda-graph-bs-decode", ["1", "2", "4"], "cuda_graph_bs_decode"),
        ("piecewise-cuda-graph-tokens", "cuda-graph-bs-prefill", ["8", "16"], "cuda_graph_bs_prefill"),
        ("piecewise-cuda-graph-max-tokens", "cuda-graph-max-bs-prefill", ["128"], "cuda_graph_max_bs_prefill"),
        ("piecewise-cuda-graph-compiler", "cuda-graph-tc-compiler", ["eager"], "cuda_graph_tc_compiler"),
        ("nsa-prefill-backend", "dsa-prefill-backend", ["flashmla_sparse"], "dsa_prefill_backend"),
        ("nsa-decode-backend", "dsa-decode-backend", ["flashmla_kv"], "dsa_decode_backend"),
        ("mamba-scheduler-strategy", "mamba-radix-cache-strategy", ["extra_buffer"], "mamba_radix_cache_strategy"),
        ("fp8-gemm-runner-backend", "fp8-gemm-backend", ["deep_gemm"], "fp8_gemm_runner_backend"),
        ("fp4-gemm-runner-backend", "fp4-gemm-backend", ["auto"], "fp4_gemm_runner_backend"),
    ],
)
def test_existing_script_flags_retain_their_values(legacy, current, values, dest):
    parser = ArgumentParser()
    add_sglang_arguments(parser)
    legacy_args = parser.parse_args([f"--sglang-{legacy}", *values])
    current_args = parser.parse_args([f"--sglang-{current}", *values])
    assert getattr(legacy_args, f"sglang_{dest}") == getattr(current_args, f"sglang_{dest}")


@pytest.mark.parametrize(
    "flag,backend",
    [
        ("enable-breakable-cuda-graph", "breakable"),
        ("disable-piecewise-cuda-graph", "disabled"),
        ("enforce-piecewise-cuda-graph", "tc_piecewise"),
    ],
)
def test_legacy_prefill_graph_switches_select_the_backend(flag, backend):
    parser = ArgumentParser()
    add_sglang_arguments(parser)
    args = parser.parse_args([f"--sglang-{flag}"])
    assert args.sglang_cuda_graph_backend_prefill == backend
    args = parser.parse_args([f"--sglang-{flag}", "--sglang-cuda-graph-backend-prefill", "disabled"])
    assert args.sglang_cuda_graph_backend_prefill == "disabled"


def test_legacy_external_pd_balancing_flag_is_accepted():
    from sglang.srt.server_args import ServerArgs

    parser = ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.parse_args(["--model-path", "model", "--prefill-round-robin-balance"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
