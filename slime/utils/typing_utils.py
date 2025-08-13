import inspect


def get_function_num_args(x):
    if inspect.isclass(x):
        func = x.__init__
    else:
        func = x

    sig = inspect.signature(func)
    required = 0
    optional = 0

    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            required += 1
        elif p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            optional += 1

    return required + optional
