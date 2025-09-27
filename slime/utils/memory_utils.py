import gc
import os
import time
import threading
import logging
from pathlib import Path
import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def clear_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def available_memory():
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return {
        "gpu": str(torch.cuda.current_device()),
        "total_GB": round(total / (1024**3), 2),
        "free_GB": round(free / (1024**3), 2),
        "used_GB": round((total - free) / (1024**3), 2),
    }


def print_memory(msg):
    if dist.get_rank() == 0:
        print(f"Memory-Usage {msg}:", available_memory())


def enable_memory_visualize(config=None):
    """Enable memory visualization using PyTorch profiler."""
    if not hasattr(torch.cuda.memory, '_record_memory_history'):
        logger.warning("Memory visualization not supported in this PyTorch version")
        return
    
    # Apply config if provided
    max_entries = 100000
    if config and hasattr(config, 'torch_memory'):
        max_entries = config.torch_memory.trace_alloc_max_entries
    
    try:
        # Try minimal API that should work across PyTorch versions
        torch.cuda.memory._record_memory_history(enabled=True)
        logger.info("Memory visualization enabled (minimal API)")
    except Exception as e:
        logger.warning(f"Failed to enable memory visualization: {e}")
        return


def dump_memory_snapshot(out_dir: str, prefix: str = "memory_snapshot"):
    """Dump memory snapshot for visualization."""
    if not hasattr(torch.cuda.memory, '_dump_snapshot'):
        logger.warning("Memory snapshot not supported in this PyTorch version")
        return
    
    # Ensure memory recording is enabled in this process
    try:
        torch.cuda.memory._record_memory_history(enabled=True)
    except Exception as e:
        logger.warning(f"Failed to enable memory recording in this process: {e}")
        
    os.makedirs(out_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    timestamp = int(time.time())
    filename = f"{prefix}_rank{rank}_{timestamp}.pickle"
    filepath = os.path.join(out_dir, filename)
    
    try:
        torch.cuda.memory._dump_snapshot(filepath)
        logger.info(f"Memory snapshot saved to {filepath}")
        
        # Also log current memory usage for immediate feedback
        mem_info = available_memory()
        logger.info(f"Current memory at {prefix}: {mem_info}")
    except Exception as e:
        # If snapshot fails, just log memory usage instead
        logger.warning(f"Failed to dump memory snapshot: {e}")
        try:
            mem_info = available_memory()
            logger.info(f"Memory usage at {prefix}: {mem_info}")
        except Exception:
            pass


class MemorySnapshotSampler:
    """Background thread to periodically sample memory snapshots."""
    
    def __init__(self, out_dir: str, interval_sec: int = 60):
        self.out_dir = out_dir
        self.interval_sec = interval_sec
        self.running = False
        self.thread = None
        
    def start(self):
        """Start periodic sampling."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        logger.info(f"Memory snapshot sampler started (interval={self.interval_sec}s, dir={self.out_dir})")
        
    def stop(self):
        """Stop periodic sampling."""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Memory snapshot sampler stopped")
        
    def _sample_loop(self):
        """Main sampling loop."""
        while self.running:
            try:
                dump_memory_snapshot(self.out_dir, "periodic")
            except Exception as e:
                logger.error(f"Error in memory sampling: {e}")
            
            time.sleep(self.interval_sec)
