"""Unit tests for SglangConfig multi-model parsing with update_weights."""

import tempfile
from argparse import Namespace

import pytest
import yaml


def _write_yaml(data: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.flush()
    return f.name


def _resolve_args() -> Namespace:
    return Namespace(rollout_num_gpus_per_engine=2, hf_checkpoint="/path/to/actor")


class TestSglangConfigUpdateWeights:
    def test_update_weights_default_true(self):
        """Models without explicit update_weights should default to True."""
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    }
                ]
            }
        )
        config = SglangConfig.from_yaml(path)
        config.models[0].resolve(_resolve_args())
        assert len(config.models) == 1
        assert config.models[0].update_weights is True

    def test_update_weights_explicit_false(self):
        """Models with update_weights: false should be parsed correctly."""
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "update_weights": True,
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "model_path": "/path/to/ref",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 2}],
                    },
                ]
            }
        )
        config = SglangConfig.from_yaml(path)
        for model in config.models:
            model.resolve(_resolve_args())
        assert len(config.models) == 2
        assert config.models[0].name == "actor"
        assert config.models[0].update_weights is True
        assert config.models[1].name == "ref"
        assert config.models[1].update_weights is False
        assert config.models[1].model_path == "/path/to/ref"

    def test_multi_model_total_gpus(self):
        """total_num_gpus should sum across all models."""
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                ]
            }
        )
        config = SglangConfig.from_yaml(path)
        assert config.total_num_gpus == 12


class TestGetModelUrl:
    def test_get_model_url_basic(self):
        """get_model_url should return the correct URL for a named model."""
        from slime.utils.url_utils import get_model_url_from_args

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers={
                "actor": ("10.0.0.1", 3000),
                "ref": ("10.0.0.1", 3001),
            },
        )
        assert get_model_url_from_args(args, "actor") == "http://10.0.0.1:3000/generate"
        assert get_model_url_from_args(args, "ref") == "http://10.0.0.1:3001/generate"
        assert get_model_url_from_args(args, "ref", "/v1/chat/completions") == "http://10.0.0.1:3001/v1/chat/completions"

    def test_get_model_url_fallback(self):
        """get_model_url should fall back to default router if model not found."""
        from slime.utils.url_utils import get_model_url_from_args

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers={"actor": ("10.0.0.1", 3000)},
        )
        assert get_model_url_from_args(args, "unknown") == "http://10.0.0.1:3000/generate"

    def test_get_model_url_no_routers(self):
        """get_model_url should work when sglang_model_routers is not set."""
        from slime.utils.url_utils import get_model_url_from_args

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
        )
        assert get_model_url_from_args(args, "anything") == "http://10.0.0.1:3000/generate"

    def test_get_model_url_router_url_precedence(self):
        """sglang_router_url should take precedence over router ip/port."""
        from slime.utils.url_utils import get_model_url_from_args

        args = Namespace(
            sglang_router_url="https://rollout.example.modal.run",
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
        )
        assert get_model_url_from_args(args, "anything") == "https://rollout.example.modal.run/generate"

    def test_get_model_url_named_url_router(self):
        """sglang_model_routers entries may be full URLs."""
        from slime.utils.url_utils import get_model_url_from_args

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers={"actor": "https://actor.example.modal.run/base"},
        )
        assert get_model_url_from_args(args, "actor") == "https://actor.example.modal.run/base/generate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
