import gc
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist


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
    memory_info = available_memory()
    if dist.get_rank() == 0:
        print(f"Memory-Usage {msg}:", memory_info)
    return memory_info


def enable_memory_visualize():
    """Enable memory history recording for debugging."""
    if hasattr(torch.cuda, "memory") and hasattr(torch.cuda.memory, "memory_snapshot"):
        try:
            torch.cuda.memory._record_memory_history(
                enabled=True, 
                alloc_trace_record_context=True, 
                alloc_trace_max_entries=100000
            )
            print("Memory visualization enabled")
        except Exception as e:
            try:
                torch.cuda.memory._record_memory_history(enabled=True)
                print("Memory visualization enabled (basic mode)")
            except Exception as e2:
                print(f"Warning: Failed to enable memory visualization: {e2}")
    else:
        print("Warning: torch.cuda.memory.memory_snapshot not available, memory visualization disabled")


def dump_memory_snapshot(tag: str = "", sub_dir: str = "", out_dir: str = "./mem_snapshots"):
    """Dump memory snapshot for debugging."""
    if not hasattr(torch.cuda, "memory") or not hasattr(torch.cuda.memory, "memory_snapshot"):
        print("Warning: memory snapshot not available")
        return
        
    rank = dist.get_rank() if dist.is_initialized() else 0
    timestamp = int(time.time())
    
    # Create output directory
    if sub_dir:
        out_path = Path(out_dir) / sub_dir
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if tag:
        filename = f"memory_snapshot_rank{rank}_{tag}_{timestamp}.pickle"
    else:
        filename = f"memory_snapshot_rank{rank}_{timestamp}.pickle"
    
    filepath = out_path / filename
    
    try:
        # Capture memory snapshot
        snapshot = torch.cuda.memory._snapshot()
        
        # Save snapshot
        with open(filepath, "wb") as f:
            pickle.dump(snapshot, f)
            
        print(f"Memory snapshot saved to {filepath}")
        
        # Print current memory usage
        memory_info = available_memory()
        print(f"Current memory usage at snapshot {tag}: {memory_info}")
        
    except Exception as e:
        print(f"Failed to dump memory snapshot: {e}")


class MemorySnapshotSampler:
    """Periodic memory snapshot sampler for debugging memory leaks."""
    
    def __init__(
        self, 
        out_dir: str = "./mem_snapshots", 
        interval: int = 100,
        enabled: bool = False
    ):
        self.out_dir = out_dir
        self.interval = interval
        self.enabled = enabled
        self.step_count = 0
        
        if self.enabled:
            enable_memory_visualize()
    
    def maybe_dump_snapshot(self, tag: str = "", force: bool = False):
        """Dump memory snapshot if conditions are met."""
        if not self.enabled:
            return
            
        self.step_count += 1
        
        if force or (self.interval > 0 and self.step_count % self.interval == 0):
            sub_dir = f"step_{self.step_count}"
            dump_tag = f"{tag}_step{self.step_count}" if tag else f"step{self.step_count}"
            dump_memory_snapshot(
                tag=dump_tag,
                sub_dir=sub_dir,
                out_dir=self.out_dir
            )
    
    def dump_at_key_points(self, point_name: str):
        """Dump snapshot at key training points."""
        if not self.enabled:
            return
            
        dump_memory_snapshot(
            tag=f"{point_name}_step{self.step_count}",
            sub_dir=f"key_points",
            out_dir=self.out_dir
        )
