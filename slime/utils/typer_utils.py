import dataclasses
import inspect
from typing import Annotated

import typer


def dataclass_cli(func, env_var_prefix: str = "SLIME_SCRIPT_"):
    """Modified from https://github.com/fastapi/typer/issues/154#issuecomment-1544876144"""

    # The dataclass type is the first argument of the function.
    sig = inspect.signature(func)
    param = list(sig.parameters.values())[0]
    dataclass_cls = param.annotation
    assert dataclasses.is_dataclass(dataclass_cls)

    # To construct the signature, we remove the first argument (self)
    # from the dataclass __init__ signature.
    signature = inspect.signature(dataclass_cls.__init__)
    old_parameters = list(signature.parameters.values())
    if len(old_parameters) > 0 and old_parameters[0].name == "self":
        del old_parameters[0]

    new_parameters = []
    for param in old_parameters:
        env_var_name = f"{env_var_prefix}{param.name.upper()}"
        new_annotation = Annotated[param.annotation, typer.Option(envvar=env_var_name)]
        new_parameters.append(param.replace(annotation=new_annotation))

    def wrapped(**kwargs):
        data = dataclass_cls(**kwargs)
        print(f"Execute command with args: {data}")
        return func(data)

    wrapped.__signature__ = signature.replace(parameters=new_parameters)
    wrapped.__doc__ = func.__doc__
    wrapped.__name__ = func.__name__
    wrapped.__qualname__ = func.__qualname__

    return wrapped


# unit test
if __name__ == "__main__":
    from typer.testing import CliRunner

    @dataclasses.dataclass
    class DemoArgs:
        name: str
        count: int = 1

    app = typer.Typer()

    @app.command()
    @dataclass_cli
    def main(args: DemoArgs):
        print(f"{args.name}|{args.count}")

    runner = CliRunner()

    res1 = runner.invoke(app, [], env={"SLIME_SCRIPT_NAME": "EnvName", "SLIME_SCRIPT_COUNT": "10"})
    print(f"{res1.stdout=}")
    assert res1.exit_code == 0
    assert "EnvName|10" in res1.stdout.strip()

    res2 = runner.invoke(app, ["--count", "999"], env={"SLIME_SCRIPT_NAME": "EnvName"})
    print(f"{res2.stdout=}")
    assert res2.exit_code == 0
    assert "EnvName|999" in res2.stdout.strip()

    print("✅ All Tests Passed!")
