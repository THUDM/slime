"""
Profiler configuration system similar to veRL PR #3099.
Supports torch_memory profiling and memory visualization.
"""

import os
import yaml
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TorchMemoryConfig:
    """Configuration for torch memory profiling."""
    trace_alloc_max_entries: int = 100000
    stack_depth: int = 32
    context: str = "all"
    stacks: str = "all"


@dataclass  
class GlobalProfilerConfig:
    """Global profiler configuration."""
    tool: str = "torch_memory"
    save_path: str = "./mem_snapshots"
    interval_sec: int = 0 # Default: disable periodic sampling
    torch_memory: TorchMemoryConfig = None
    
    def __post_init__(self):
        if self.torch_memory is None:
            self.torch_memory = TorchMemoryConfig()


def load_profiler_config(config_path: Optional[str] = None) -> GlobalProfilerConfig:
    """Load profiler configuration from YAML file."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            # Extract profiler config if nested
            if 'global_profiler' in config_dict:
                profiler_dict = config_dict['global_profiler']
            else:
                profiler_dict = config_dict
                
            # Handle torch_memory sub-config
            torch_memory_dict = profiler_dict.pop('torch_memory', {})
            torch_memory_config = TorchMemoryConfig(**torch_memory_dict)
            
            config = GlobalProfilerConfig(**profiler_dict)
            config.torch_memory = torch_memory_config
            
            logger.info(f"Loaded profiler config from {config_path}")
            return config
            
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            
    return GlobalProfilerConfig()


def save_profiler_config(config: GlobalProfilerConfig, config_path: str):
    """Save profiler configuration to YAML file."""
    config_dict = {
        'global_profiler': asdict(config)
    }
    
    os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Saved profiler config to {config_path}")


def create_default_config() -> str:
    """Create a default profiler configuration file."""
    config = GlobalProfilerConfig(
        tool="torch_memory",
        save_path="./mem_snapshots", 
        interval_sec=30,
        enabled=True,
        torch_memory=TorchMemoryConfig(
            trace_alloc_max_entries=100000,
            stack_depth=32,
            context="all",
            stacks="all"
        )
    )
    
    config_path = "./profiler_config.yaml"
    save_profiler_config(config, config_path)
    return config_path