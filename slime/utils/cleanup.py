"""Cleanup policy shared by resource owners and the training driver."""

import logging
import time

logger = logging.getLogger(__name__)


class Cleanup:
    """Try every cleanup step, keeping the original failure and one deadline.

    Owners must pass ``remaining`` to blocking operations. Arbitrary custom
    hooks cannot be interrupted safely in a thread; the driver's bounded RPC
    and actor termination provide the outer limit for those hooks.
    """

    def __init__(self, timeout=60):
        self.deadline = time.monotonic() + timeout
        self.errors = []

    @property
    def remaining(self):
        return max(0, self.deadline - time.monotonic())

    def run(self, description, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            self.errors.append(error)
            logger.exception("Failed to %s", description)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        # During error unwinding, report cleanup failures without replacing
        # the training/startup error. Otherwise cleanup failure is observable.
        if exc is None and self.errors:
            raise self.errors[0]
