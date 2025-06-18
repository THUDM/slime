from time import time


__all__ = ["Timer", "timer"]


class Timer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timer, cls).__new__(cls)
            cls._instance.timers = {}
            cls._instance.start_time = {}
            cls._instance.seq_lens = None
        return cls._instance

    def start(self, name):
        assert name not in self.timers, f"Timer {name} already started."
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


class TimerContext:
    def __init__(self, name):
        self.name = name
        self.timer_instance = Timer()

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time()
        self.timer_instance.add(self.name, self.end_time - self.start_time)
        return False  # Don't suppress exceptions


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
        return TimerContext(name)

    func = name_or_func
    name = func.__name__

    def wrapper(*args, **kwargs):
        timer_instance = Timer()
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        timer_instance.add(name, end_time - start_time)
        return result

    return wrapper
