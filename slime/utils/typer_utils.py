import dataclasses
import inspect


def dataclass_cli(func):
    """Modified from https://github.com/fastapi/typer/issues/154#issuecomment-1544876144"""

    # The dataclass type is the first argument of the function.
    sig = inspect.signature(func)
    param = list(sig.parameters.values())[0]
    dataclass_cls = param.annotation
    assert dataclasses.is_dataclass(dataclass_cls)

    # To construct the signature, we remove the first argument (self)
    # from the dataclass __init__ signature.
    signature = inspect.signature(dataclass_cls.__init__)
    parameters = list(signature.parameters.values())
    if len(parameters) > 0 and parameters[0].name == "self":
        del parameters[0]

    def wrapped(**kwargs):
        return func(dataclass_cls(**kwargs))

    wrapped.__signature__ = signature.replace(parameters=parameters)
    wrapped.__doc__ = func.__doc__

    return wrapped
