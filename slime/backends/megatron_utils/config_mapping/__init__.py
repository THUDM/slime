from .registry import mapper_registry, register_mapper


def get_mapper(name: str):
    return mapper_registry.get_mapper(name)


__all__ = [
    "register_mapper",
    "mapper_registry",
    "get_mapper",
]
