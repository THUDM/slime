"""
Memory profiler manager similar to veRL PR #3099.
Handles torch_memory profiling and memory visualization.
"""

import os
import logging
from typing import Optional
from .profiler_config import GlobalProfilerConfig, load_profiler_config
from .memory_utils import enable_memory_visualize, MemorySnapshotSampler, dump_memory_snapshot

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Memory profiler manager for torch_memory tool."""
    
    def __init__(self, config: GlobalProfilerConfig):
        self.config = config
        self.sampler: Optional[MemorySnapshotSampler] = None
        
    def setup(self):
        """Setup memory profiling."""
        enable_memory_visualize(self.config)
        
        if self.config.interval_sec > 0:
            self.sampler = MemorySnapshotSampler(self.config.save_path, self.config.interval_sec)
            self.sampler.start()
            logger.info(f"Memory profiler setup with periodic sampling (save_path={self.config.save_path}, interval={self.config.interval_sec}s)")
        else:
            logger.info(f"Memory profiler setup with manual snapshots only (save_path={self.config.save_path})")
    
    def snapshot(self, prefix: str = "manual"):
        """Take a manual memory snapshot."""
        dump_memory_snapshot(self.config.save_path, prefix)
    
    def shutdown(self):
        """Shutdown memory profiler."""
        if self.sampler:
            self.sampler.stop()
            self.sampler = None


class GlobalProfiler:
    """Global profiler manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_profiler_config(config_path)
        self.memory_profiler: Optional[MemoryProfiler] = None
        
    def setup(self):
        """Setup profilers based on configuration."""
        if self.config.tool == "torch_memory":
            self.memory_profiler = MemoryProfiler(self.config)
            self.memory_profiler.setup()
    
    def snapshot(self, prefix: str = "manual"):
        """Take manual snapshots."""
        if self.memory_profiler:
            self.memory_profiler.snapshot(prefix)
    
    def shutdown(self):
        """Shutdown profiler."""
        if self.memory_profiler:
            self.memory_profiler.shutdown()


# Global profiler instance
_global_profiler: Optional[GlobalProfiler] = None


def init_global_profiler(config_path: Optional[str] = None) -> GlobalProfiler:
    """Initialize global profiler."""
    global _global_profiler
    _global_profiler = GlobalProfiler(config_path)
    _global_profiler.setup()
    return _global_profiler


def get_global_profiler() -> Optional[GlobalProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def snapshot_memory(prefix: str = "manual"):
    """Convenience function to take memory snapshot."""
    if _global_profiler:
        _global_profiler.snapshot(prefix)


def shutdown_global_profiler():
    """Shutdown global profiler."""
    global _global_profiler
    if _global_profiler:
        _global_profiler.shutdown()
        _global_profiler = None