import pytest

from slime.utils.external_utils.command_utils import _parse_extra_env_vars

NUM_GPUS = 0


def test_parse_extra_env_vars_preserves_equals_in_values():
    assert _parse_extra_env_vars("TOKEN=abc== END=ok") == {"TOKEN": "abc==", "END": "ok"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
