import argparse
from argparse import Namespace

import pytest

from slime.backends.megatron_utils.server.arguments import (
    add_megatron_server_arguments,
    configure_megatron_server_args,
    validate_megatron_server_args,
)

NUM_GPUS = 0


def _server_args(**overrides):
    values = dict(
        teacher_port=7999,
        teacher_warmup_port=7999,
        teacher_warmup_timeout_s=3000,
        teacher_sample_reduction_chunk_size=4096,
        teacher_label_reduction_chunk_size=4096,
        megatron_server_max_length=0,
        megatron_server_update_timeout_s=3600.0,
        megatron_server_warmup=True,
        debug_train_only=False,
        use_kl_loss=True,
        offload_train=True,
        use_dynamic_batch_size=True,
        use_wandb=True,
        kl_coef=0.1,
        use_opd=True,
        use_critic=True,
        keep_old_actor=True,
        no_load_optim=False,
        no_load_rng=False,
        only_train_params_name_list=["actor"],
    )
    values.update(overrides)
    return Namespace(**values)


def test_add_megatron_server_arguments():
    parser = argparse.ArgumentParser()
    add_megatron_server_arguments(parser)

    args = parser.parse_args(
        [
            "--teacher-port",
            "8123",
            "--teacher-sample-reduction-chunk-size",
            "128",
            "--teacher-label-reduction-chunk-size",
            "256",
            "--no-megatron-server-warmup",
        ]
    )

    assert args.teacher_port == 8123
    assert args.teacher_sample_reduction_chunk_size == 128
    assert args.teacher_label_reduction_chunk_size == 256
    assert args.megatron_server_warmup is False
    assert args.teacher_warmup_timeout_s == 3000
    assert args.megatron_server_update_timeout_s == 3600.0


def test_configure_megatron_server_args_forces_teacher_only_mode():
    args = configure_megatron_server_args(_server_args())
    validate_megatron_server_args(args)

    assert args.debug_train_only is True
    assert args.use_kl_loss is False
    assert args.use_opd is False
    assert args.use_critic is False
    assert args.keep_old_actor is False
    assert args.no_load_optim is True
    assert args.no_load_rng is True
    assert args.only_train_params_name_list == ["nothing_to_train"]


def test_validate_megatron_server_args_rejects_invalid_server_values():
    args = configure_megatron_server_args(_server_args(teacher_sample_reduction_chunk_size=0))

    with pytest.raises(ValueError, match="teacher_sample_reduction_chunk_size"):
        validate_megatron_server_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
