from contextlib import contextmanager
from functools import wraps
from time import time

from .misc import SingletonMeta

__all__ = ["Timer", "timer"]


class Timer(metaclass=SingletonMeta):
    def __init__(self):
        self.timers = {}
        self.start_time = {}

    def start(self, name):
        assert name not in self.start_time, f"Timer {name} already started."
        self.start_time[name] = time()

    def end(self, name):
        assert name in self.start_time, f"Timer {name} not started."
        elapsed_time = time() - self.start_time[name]
        self.add(name, elapsed_time)
        del self.start_time[name]

    def reset(self, name=None):
        if name is None:
            self.timers = {}
        elif name in self.timers:
            del self.timers[name]

    def add(self, name, elapsed_time):
        if name not in self.timers:
            self.timers[name] = elapsed_time
        else:
            self.timers[name] += elapsed_time

    def log_dict(self):
        return self.timers

    @contextmanager
    def context(self, name):
        self.start(name)
        try:
            yield
        finally:
            self.end(name)


def timer(name_or_func):
    """
    Can be used either as a decorator or a context manager:

    @timer
    def func():
        ...

    or

    with timer("block_name"):
        ...
    """
    # When used as a context manager
    if isinstance(name_or_func, str):
        name = name_or_func
        return Timer().context(name)

    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer().context(func.__name__):
            return func(*args, **kwargs)

    return wrapper
