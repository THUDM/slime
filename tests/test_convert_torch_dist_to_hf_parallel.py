import importlib.util
from pathlib import Path

import safetensors.torch
import torch


CONVERTER_PATH = Path(__file__).parents[1] / "tools" / "convert_torch_dist_to_hf_parallel.py"
SPEC = importlib.util.spec_from_file_location("convert_torch_dist_to_hf_parallel", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)
save_missing_tensors = CONVERTER.save_missing_tensors


def _load_output_tensor(output_dir, weight_map, name):
    return safetensors.torch.load_file(output_dir / weight_map[name])[name]


def test_save_missing_tensors_only_copies_unconverted_weights(tmp_path):
    origin_dir = tmp_path / "origin"
    output_dir = tmp_path / "output"
    origin_dir.mkdir()
    output_dir.mkdir()

    converted = torch.tensor([1.0, 2.0])
    missing_visual = torch.tensor([3.0, 4.0])
    missing_mtp = torch.tensor([5.0, 6.0, 7.0])
    safetensors.torch.save_file(
        {"model.converted.weight": converted, "model.visual.weight": missing_visual},
        origin_dir / "model-00001-of-00002.safetensors",
    )
    safetensors.torch.save_file(
        {"mtp.weight": missing_mtp},
        origin_dir / "model-00002-of-00002.safetensors",
    )

    weight_map, total_size, next_file_index = save_missing_tensors(
        origin_dir,
        {"model.converted.weight"},
        output_dir,
        chunk_size=12,
        start_file_index=3,
    )

    assert set(weight_map) == {"model.visual.weight", "mtp.weight"}
    assert total_size == (
        missing_visual.numel() * missing_visual.element_size() + missing_mtp.numel() * missing_mtp.element_size()
    )
    assert next_file_index == 5
    assert sorted(path.name for path in output_dir.glob("*.safetensors")) == [
        "model-00003.safetensors",
        "model-00004.safetensors",
    ]
    torch.testing.assert_close(_load_output_tensor(output_dir, weight_map, "model.visual.weight"), missing_visual)
    torch.testing.assert_close(_load_output_tensor(output_dir, weight_map, "mtp.weight"), missing_mtp)


def test_save_missing_tensors_writes_nothing_when_checkpoint_is_complete(tmp_path):
    origin_dir = tmp_path / "origin"
    output_dir = tmp_path / "output"
    origin_dir.mkdir()
    output_dir.mkdir()
    tensor = torch.tensor([1.0, 2.0])
    safetensors.torch.save_file({"model.weight": tensor}, origin_dir / "model.safetensors")

    weight_map, total_size, next_file_index = save_missing_tensors(
        origin_dir,
        {"model.weight"},
        output_dir,
        chunk_size=1024,
        start_file_index=2,
    )

    assert weight_map == {}
    assert total_size == 0
    assert next_file_index == 2
    assert list(output_dir.iterdir()) == []
