from argparse import Namespace

import pytest

from slime.backends.sglang_utils.arguments import validate_args

NUM_GPUS = 0


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
