import importlib
import sys
import types

fake_typer = types.ModuleType("typer")
fake_typer.Option = lambda *args, **kwargs: None
sys.modules.setdefault("typer", fake_typer)

fake_misc = types.ModuleType("slime.utils.misc")
fake_misc.exec_command = lambda *args, **kwargs: None
sys.modules["slime.utils.misc"] = fake_misc

U = importlib.import_module("slime.utils.external_utils.command_utils")


def test_convert_checkpoint_keeps_extra_args_separated(monkeypatch, tmp_path):
    captured = []

    def fake_exec_command(cmd: str, capture_output: bool = False):
        captured.append(cmd)
        return ""

    monkeypatch.setattr(U, "exec_command", fake_exec_command)

    U.convert_checkpoint(
        model_name="demo-model",
        megatron_model_type="qwen2.5-0.5B",
        num_gpus_per_node=1,
        dir_dst=str(tmp_path),
        extra_args="--disable-bias-linear --untie-embeddings-and-output-weights",
    )

    assert len(captured) == 1
    cmd = captured[0]
    save_arg = f"--save {tmp_path}/demo-model_torch_dist"
    assert f"{save_arg} --disable-bias-linear --untie-embeddings-and-output-weights" in cmd
