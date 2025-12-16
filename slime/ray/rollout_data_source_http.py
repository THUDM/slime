"""
HTTP-based Rollout Data Source with Buffer for Agent Tasks

This module provides an HTTP-based buffer implementation that connects to
the independent rollout buffer server (slime_plugins/rollout_buffer/buffer.py).

Architecture:
    RolloutManager
        ↓
    RolloutDataSourceWithHTTPBuffer (this file)
        ↓ HTTP API
    Rollout Buffer Server (slime_plugins/rollout_buffer/buffer.py)
        ↓
    Agent Framework / Custom Generators

Usage:
    # In your training script args
    --buffer_mode http
    --buffer_server_url http://localhost:8889
    --buffer_task_type grpo  # or your custom agent task

Comparison with In-Process Buffer:
    - In-Process: Fast, embedded, for standard GRPO/PPO
    - HTTP: Flexible, scalable, for agent tasks and async generation

"""

import time
from typing import Optional, Dict, Any, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from slime.ray.rollout_data_source import RolloutDataSource
from slime.utils.types import Sample


class HTTPBufferClient:
    """
    HTTP client for communicating with the rollout buffer server.

    Handles connection pooling, retries, and serialization/deserialization.
    """

    def __init__(self, server_url: str, timeout: int = 30, max_retries: int = 3):
        """
        Initialize HTTP buffer client.

        Args:
            server_url: Base URL of the buffer server (e.g., http://localhost:8889)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout

        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        print(f"[HTTP Buffer Client] Initialized: server={server_url}, timeout={timeout}s")

    def write(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write a single sample to the buffer.

        Args:
            data: Sample data in HTTP format (dict)

        Returns:
            Response from server
        """
        url = f"{self.server_url}/buffer/write"
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[HTTP Buffer Client] Write failed: {e}")
            raise

    def read(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Read samples from the buffer.

        Args:
            num_samples: Number of samples to read (optional, server decides)

        Returns:
            Response containing samples and metadata
        """
        url = f"{self.server_url}/get_rollout_data"
        payload = {"num_samples": num_samples} if num_samples else {}
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[HTTP Buffer Client] Read failed: {e}")
            raise

    def start_rollout(self, task_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start rollout generation on the buffer server.

        Args:
            task_type: Type of task (e.g., "grpo", "math", "tool")
            config: Configuration for the generator

        Returns:
            Response from server
        """
        url = f"{self.server_url}/start_rollout"
        payload = {
            "task_type": task_type,
            **config
        }
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[HTTP Buffer Client] Start rollout failed: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if the buffer server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = self.session.get(f"{self.server_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False


class RolloutDataSourceWithHTTPBuffer(RolloutDataSource):
    """
    HTTP-based rollout data source with buffer support.

    This implementation connects to an independent HTTP buffer server for:
    - Agent-based RL tasks
    - Asynchronous trajectory generation
    - Scalable buffer management
    - Multi-task support

    Features:
    - Automatic format conversion (Sample ↔ HTTP dict)
    - Policy version tracking for off-policy GRPO
    - Auto-detection of on-policy/off-policy mode
    - Compatible with existing training pipeline

    Configuration:
        --buffer_mode http
        --buffer_server_url http://localhost:8889
        --buffer_task_type grpo
        --use_buffer True/False (optional, auto-detect from loss_type)
        --buffer_max_size 1000
    """

    def __init__(self, args):
        super().__init__(args)

        # === HTTP Buffer Configuration ===
        self.buffer_server_url = getattr(args, 'buffer_server_url', 'http://localhost:8889')
        self.buffer_task_type = getattr(args, 'buffer_task_type', 'grpo')
        self.buffer_enabled = getattr(args, 'use_buffer', None)

        # === Auto-detect buffer usage based on loss_type ===
        if self.buffer_enabled is None:
            loss_type = getattr(args, 'loss_type', 'policy_loss')
            self.buffer_enabled = (loss_type == 'decoupled_policy_loss')

            if self.buffer_enabled:
                print(f"[HTTP Buffer] Auto-enabled for off-policy GRPO (loss_type={loss_type})")
            else:
                print(f"[HTTP Buffer] Disabled for on-policy GRPO (loss_type={loss_type})")

        # === Policy Version Tracking ===
        self.current_policy_version = 0

        # === HTTP Client ===
        if self.buffer_enabled:
            self.http_client = HTTPBufferClient(
                server_url=self.buffer_server_url,
                timeout=getattr(args, 'buffer_timeout', 30),
                max_retries=getattr(args, 'buffer_max_retries', 3)
            )

            # Health check
            if not self.http_client.health_check():
                print(f"[HTTP Buffer] WARNING: Server at {self.buffer_server_url} is not responding!")
                print(f"[HTTP Buffer] Make sure to start the buffer server first:")
                print(f"[HTTP Buffer]   cd slime_plugins/rollout_buffer && python buffer.py")

        # === Statistics ===
        self.total_added = 0
        self.total_sampled = 0

        print(f"[HTTP Buffer] Initialized: "
              f"enabled={self.buffer_enabled}, "
              f"server={self.buffer_server_url}, "
              f"task_type={self.buffer_task_type}")

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Get samples with automatic buffer mixing.

        Behavior:
        - If buffer disabled: return only new samples from dataset (on-policy)
        - If buffer enabled: fetch from HTTP buffer server (off-policy)

        Args:
            num_samples: Number of sample groups to retrieve

        Returns:
            List of sample groups
        """
        if not self.buffer_enabled:
            # Buffer disabled: use only new data (on-policy behavior)
            return super().get_samples(num_samples=num_samples)

        # Buffer enabled: fetch from HTTP server
        try:
            response = self.http_client.read(num_samples=num_samples * self.args.n_samples_per_prompt)

            if not response.get("success", False):
                print(f"[HTTP Buffer] Read failed, falling back to dataset")
                return super().get_samples(num_samples=num_samples)

            data = response.get("data", {}).get("data", [])

            if not data:
                print(f"[HTTP Buffer] No data available, falling back to dataset")
                return super().get_samples(num_samples=num_samples)

            # Convert HTTP response to Sample objects
            samples = self._convert_http_to_samples(data)

            # Group samples by n_samples_per_prompt
            grouped_samples = []
            for i in range(0, len(samples), self.args.n_samples_per_prompt):
                group = samples[i:i + self.args.n_samples_per_prompt]
                if len(group) == self.args.n_samples_per_prompt:
                    grouped_samples.append(group)

            self.total_sampled += len(grouped_samples)

            print(f"[HTTP Buffer] Fetched {len(grouped_samples)} groups ({len(samples)} samples)")

            return grouped_samples

        except Exception as e:
            print(f"[HTTP Buffer] Error fetching data: {e}, falling back to dataset")
            return super().get_samples(num_samples=num_samples)

    def add_samples(self, samples: List[Sample]):
        """
        Add samples to HTTP buffer.

        Automatically converts Sample objects to HTTP format and sends to server.

        Args:
            samples: List of samples (flat list)
        """
        if not samples:
            return

        if not self.buffer_enabled:
            # Buffer disabled: do nothing (on-policy - use data once)
            return

        try:
            # Convert samples to HTTP format
            http_data = self._convert_samples_to_http(samples)

            # Write to HTTP buffer
            for item in http_data:
                self.http_client.write(item)
                self.total_added += 1

            print(f"[HTTP Buffer] Added {len(http_data)} samples to server")

        except Exception as e:
            print(f"[HTTP Buffer] Error adding samples: {e}")
            # Don't fail the training, just skip buffer

    def _convert_samples_to_http(self, samples: List[Sample]) -> List[Dict[str, Any]]:
        """
        Convert Sample objects to HTTP buffer format.

        Args:
            samples: List of Sample objects

        Returns:
            List of dicts in HTTP format
        """
        http_data = []
        none_reward_count = 0
        none_policy_version_count = 0

        for idx, sample in enumerate(samples):
            # Create instance_id from group_index
            instance_id = f"grpo_group_{sample.group_index if hasattr(sample, 'group_index') else idx}"

            # === Robust reward handling ===
            reward = sample.reward
            if reward is None:
                none_reward_count += 1
                reward = 0.0

            # === Robust policy_version handling ===
            policy_version = getattr(sample, 'policy_version', None)
            if policy_version is None:
                none_policy_version_count += 1
                policy_version = self.current_policy_version

            item = {
                "instance_id": instance_id,
                "prompt": sample.prompt,
                "response": sample.response,
                "reward": reward,
                "policy_version": policy_version,
                "response_length": sample.response_length,
                "timestamp": time.time(),
                # Additional fields for reconstruction
                "tokens": sample.tokens,
                "loss_mask": sample.loss_mask if hasattr(sample, 'loss_mask') else None,
                "index": sample.index if hasattr(sample, 'index') else idx,
            }

            http_data.append(item)

        # Log warnings for data quality issues
        if none_reward_count > 0:
            print(f"[HTTP Buffer] WARNING: {none_reward_count}/{len(samples)} samples had None reward, converted to 0.0")
        if none_policy_version_count > 0:
            print(f"[HTTP Buffer] WARNING: {none_policy_version_count}/{len(samples)} samples had None policy_version, "
                  f"using current_policy_version={self.current_policy_version}")

        return http_data

    def _convert_http_to_samples(self, http_data: List[Dict[str, Any]]) -> List[Sample]:
        """
        Convert HTTP format back to Sample objects.

        Args:
            http_data: List of dicts from HTTP response

        Returns:
            List of Sample objects
        """
        samples = []

        for idx, item in enumerate(http_data):
            # === Robust reward handling ===
            # Check if reward exists and is valid
            reward = item.get("reward", None)
            if reward is None:
                print(f"[HTTP Buffer] WARNING: Sample {idx} has None reward, using 0.0")
                reward = 0.0

            sample = Sample(
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                reward=reward,
                response_length=item.get("response_length", 0),
                tokens=item.get("tokens", []),
            )

            # === Robust policy_version handling ===
            # For off-policy GRPO, policy_version is critical
            # Always set it to a valid value (never None)
            if "policy_version" in item and item["policy_version"] is not None:
                sample.policy_version = item["policy_version"]
            else:
                # Use current policy version as fallback
                sample.policy_version = self.current_policy_version
                if idx == 0:  # Only log once per batch
                    print(f"[HTTP Buffer] WARNING: policy_version missing in HTTP data, "
                          f"using current_policy_version={self.current_policy_version}")

            # Restore other optional fields
            if "loss_mask" in item and item["loss_mask"] is not None:
                sample.loss_mask = item["loss_mask"]
            if "index" in item:
                sample.index = item["index"]
            if "group_index" in item:
                sample.group_index = item["group_index"]

            samples.append(sample)

        return samples

    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics from HTTP server.

        Returns:
            Dictionary with buffer metrics
        """
        if not self.buffer_enabled:
            return {
                "enabled": False,
                "mode": "http",
                "buffer_size": 0,
            }

        stats = {
            "enabled": True,
            "mode": "http",
            "server_url": self.buffer_server_url,
            "task_type": self.buffer_task_type,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "current_policy_version": self.current_policy_version,
        }

        # Try to get server-side stats
        try:
            # This would require adding a stats endpoint to the buffer server
            # For now, return client-side stats
            pass
        except:
            pass

        return stats

    def update_policy_version(self, version: int):
        """Update current policy version for staleness-aware sampling"""
        self.current_policy_version = version
        print(f"[HTTP Buffer] Policy version updated to {version}")

    def start_async_generation(self, config: Dict[str, Any]):
        """
        Start asynchronous trajectory generation on the buffer server.

        This is useful for agent tasks where generation can happen in parallel
        with training.

        Args:
            config: Configuration for the generator
        """
        if not self.buffer_enabled:
            return

        try:
            response = self.http_client.start_rollout(
                task_type=self.buffer_task_type,
                config=config
            )
            print(f"[HTTP Buffer] Started async generation: {response.get('message', 'OK')}")
        except Exception as e:
            print(f"[HTTP Buffer] Failed to start async generation: {e}")
