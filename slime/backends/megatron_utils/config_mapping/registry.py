import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class MapperRegistry:
    """
    Registry for config mappers.
    """

    def __init__(self):
        self._mappers: dict[str, Callable] = {}

    def register(self, model_types: list[str], mapper_func: Callable):
        if not callable(mapper_func):
            raise TypeError(f"Mapper for {model_types} must be callable")

        for name in model_types:
            if name in self._mappers:
                logger.warning(f"Mapper for {name} is being overridden")
            self._mappers[name] = mapper_func
            logger.info(f"Registered config mapper for model type: {name}")

    def get_mapper(self, name: str) -> Callable:
        """
        Get the mapper by model_type.
        """
        if name not in self._mappers:
            raise ValueError(f"Mapper for {name} is not registered.")
        return self._mappers[name]

    def list_registered_mappers(self) -> list[str]:
        return list(self._mappers.keys())


# Global registry instance
mapper_registry = MapperRegistry()


def register_mapper(*args):
    """
    Decorator: register config mapper.

    Args: suppotred model_types.
    """

    def decorator(func: Callable):
        mapper_registry.register(
            model_types=list(args),
            mapper_func=func,
        )
        return func

    return decorator
