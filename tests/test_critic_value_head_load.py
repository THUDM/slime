import ast
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0
MODEL_PATH = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/model.py"


def _load_hide_output_layer_function(output_layer):
    tree = ast.parse(MODEL_PATH.read_text())
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_hide_critic_output_layer_during_policy_checkpoint_load"
    )
    function_node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function_node], type_ignores=[]))
    namespace = {
        "DDP": object,
        "Sequence": Sequence,
        "_iter_critic_output_layers": lambda _model: [(0, output_layer)],
    }
    exec(compile(module, str(MODEL_PATH), "exec"), namespace)
    return contextmanager(namespace["_hide_critic_output_layer_during_policy_checkpoint_load"])


@pytest.mark.unit
def test_policy_checkpoint_load_temporarily_hides_full_critic_output_layer():
    output_layer = torch.nn.Linear(4, 1, bias=True)
    original_weight = output_layer.weight
    original_bias = output_layer.bias
    optimizer = torch.optim.SGD(output_layer.parameters(), lr=0.1)
    hide_output_layer = _load_hide_output_layer_function(output_layer)

    with hide_output_layer([object()], enabled=True):
        assert output_layer.weight is None
        assert output_layer.bias is None
        assert dict(output_layer.named_parameters()) == {}

    assert output_layer.weight is original_weight
    assert output_layer.bias is original_bias
    assert optimizer.param_groups[0]["params"][0] is original_weight
    assert optimizer.param_groups[0]["params"][1] is original_bias


@pytest.mark.unit
def test_policy_checkpoint_load_restores_critic_output_layer_after_failure():
    output_layer = torch.nn.Linear(4, 1, bias=True)
    original_weight = output_layer.weight
    original_bias = output_layer.bias
    hide_output_layer = _load_hide_output_layer_function(output_layer)

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        with hide_output_layer([object()], enabled=True):
            raise RuntimeError("checkpoint load failed")

    assert output_layer.weight is original_weight
    assert output_layer.bias is original_bias
