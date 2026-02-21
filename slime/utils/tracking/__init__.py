"""
Shared tracking interface for experiment logging backends.

Exports :class:`TrackingManager` so existing ``from .tracking import TrackingManager``
imports continue to work.
"""

from .manager import TrackingBackend, TrackingManager  # noqa: F401
from .mlflow_utils import finish as mlflow_finish  # noqa: F401
from .mlflow_utils import init_mlflow, log_metrics  # noqa: F401
