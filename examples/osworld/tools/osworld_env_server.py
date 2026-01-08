#!/usr/bin/env python3
"""HTTP server that exposes OSWorld DesktopEnv for remote access.

Run on HOST machine (in osworld_venv) to bridge OSWorld VMs to Slime container.

Usage:
    cd ~/OSWorld
    source ~/osworld_venv/bin/activate
    python osworld_env_server.py --port 8100

The Slime container connects via OSWORLD_SERVER_URL=http://172.17.0.1:8100
"""

import argparse
import base64
import io
import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Lazy import - only load when needed
_DesktopEnv = None


def get_desktop_env_class():
    """Lazy import DesktopEnv to avoid import errors on startup."""
    global _DesktopEnv
    if _DesktopEnv is None:
        from desktop_env.desktop_env import DesktopEnv

        _DesktopEnv = DesktopEnv
    return _DesktopEnv


class Episode:
    """Manages a single OSWorld episode lifecycle."""

    def __init__(self, episode_id: str, env_config: dict):
        self.episode_id = episode_id
        self.env_config = env_config
        self.env = None
        self.step_count = 0
        self.task_config = None
        self.last_obs = None
        self.created_at = time.time()
        self.lock = threading.Lock()

    def initialize(self, task_config: dict) -> dict:
        """Create DesktopEnv and reset with task config."""
        with self.lock:
            self.task_config = task_config
            self.step_count = 0

            DesktopEnv = get_desktop_env_class()
            self.env = DesktopEnv(
                provider_name=self.env_config.get("provider_name", "docker"),
                os_type=self.env_config.get("os_type", "Ubuntu"),
                action_space=self.env_config.get("action_space", "pyautogui"),
                screen_size=tuple(self.env_config.get("screen_size", [1920, 1080])),
                headless=self.env_config.get("headless", True),
                require_a11y_tree=self.env_config.get("require_a11y_tree", True),
            )

            obs = self.env.reset(task_config=task_config)
            self.last_obs = obs
            return self._encode_observation(obs)

    def step(self, action: str, pause: int = 2) -> dict:
        """Execute action and return new observation."""
        with self.lock:
            if self.env is None:
                raise RuntimeError("Episode not initialized")

            obs, reward, done, info = self.env.step(action, pause=pause)
            self.step_count += 1
            self.last_obs = obs

            result = self._encode_observation(obs)
            result.update(
                {
                    "reward": float(reward),
                    "done": bool(done),
                    "step": self.step_count,
                    "info": info,
                }
            )
            return result

    def render(self) -> dict:
        """Get current observation without stepping."""
        with self.lock:
            if self.last_obs is None:
                return {"error": "No observation available"}
            return self._encode_observation(self.last_obs)

    def evaluate(self) -> dict:
        """Compute task completion reward."""
        with self.lock:
            if self.env is None:
                return {"reward": 0.0, "error": "Episode not initialized"}

            try:
                reward = float(self.env.evaluate())
                return {"reward": reward}
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                return {"reward": 0.0, "error": str(e)}

    def close(self):
        """Clean up environment resources."""
        with self.lock:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception as e:
                    logger.warning(f"Error closing env: {e}")
                self.env = None

    def _encode_observation(self, obs: dict) -> dict:
        """Convert observation to JSON-serializable format."""
        result = {
            "accessibility_tree": obs.get("accessibility_tree", ""),
            "terminal": obs.get("terminal", ""),
            "instruction": obs.get("instruction", ""),
        }

        screenshot = obs.get("screenshot")
        if screenshot is not None:
            if isinstance(screenshot, bytes):
                result["screenshot_base64"] = base64.b64encode(screenshot).decode("utf-8")
            elif hasattr(screenshot, "tobytes"):
                # PIL Image
                buf = io.BytesIO()
                screenshot.save(buf, format="PNG")
                result["screenshot_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        return result


class EpisodeManager:
    """Thread-safe episode management."""

    def __init__(self, max_episodes: int = 10, ttl_seconds: int = 3600):
        self.episodes: dict[str, Episode] = {}
        self.max_episodes = max_episodes
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()

    def create(self, env_config: dict) -> str:
        """Create new episode and return ID."""
        with self.lock:
            self._cleanup_stale()

            if len(self.episodes) >= self.max_episodes:
                # Remove oldest
                oldest_id = min(self.episodes.keys(), key=lambda k: self.episodes[k].created_at)
                self.episodes[oldest_id].close()
                del self.episodes[oldest_id]

            episode_id = str(uuid.uuid4())[:8]
            self.episodes[episode_id] = Episode(episode_id, env_config)
            return episode_id

    def get(self, episode_id: str) -> Episode | None:
        """Get episode by ID."""
        with self.lock:
            return self.episodes.get(episode_id)

    def remove(self, episode_id: str):
        """Remove and close episode."""
        with self.lock:
            if episode_id in self.episodes:
                self.episodes[episode_id].close()
                del self.episodes[episode_id]

    def _cleanup_stale(self):
        """Remove episodes older than TTL."""
        now = time.time()
        stale = [eid for eid, ep in self.episodes.items() if now - ep.created_at > self.ttl_seconds]
        for eid in stale:
            self.episodes[eid].close()
            del self.episodes[eid]
            logger.info(f"Cleaned up stale episode: {eid}")


class OSWorldHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OSWorld endpoints."""

    manager: EpisodeManager = None  # Set by server

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")

    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        """Read JSON request body."""
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json({"status": "ok", "episodes": len(self.manager.episodes)})
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        try:
            data = self._read_json()

            if self.path == "/reset":
                self._handle_reset(data)
            elif self.path == "/step":
                self._handle_step(data)
            elif self.path == "/render":
                self._handle_render(data)
            elif self.path == "/evaluate":
                self._handle_evaluate(data)
            elif self.path == "/close":
                self._handle_close(data)
            else:
                self._send_json({"error": "Not found"}, 404)

        except Exception as e:
            logger.exception(f"Request error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _handle_reset(self, data: dict):
        """Initialize new episode."""
        task_config = data.get("task_config", {})
        env_config = data.get("env_config", {})

        episode_id = self.manager.create(env_config)
        episode = self.manager.get(episode_id)

        logger.info(f"Creating episode {episode_id} for task: {task_config.get('id', 'unknown')}")

        try:
            result = episode.initialize(task_config)
            result["episode_id"] = episode_id
            self._send_json(result)
        except Exception:
            self.manager.remove(episode_id)
            raise

    def _handle_step(self, data: dict):
        """Execute action in episode."""
        episode_id = data.get("episode_id")
        action = data.get("action", "")
        pause = data.get("pause", 2)

        episode = self.manager.get(episode_id)
        if episode is None:
            self._send_json({"error": f"Episode not found: {episode_id}"}, 404)
            return

        result = episode.step(action, pause)
        self._send_json(result)

    def _handle_render(self, data: dict):
        """Get current observation."""
        episode_id = data.get("episode_id")
        episode = self.manager.get(episode_id)

        if episode is None:
            self._send_json({"error": f"Episode not found: {episode_id}"}, 404)
            return

        result = episode.render()
        self._send_json(result)

    def _handle_evaluate(self, data: dict):
        """Compute task reward."""
        episode_id = data.get("episode_id")
        episode = self.manager.get(episode_id)

        if episode is None:
            self._send_json({"error": f"Episode not found: {episode_id}"}, 404)
            return

        result = episode.evaluate()
        self._send_json(result)

    def _handle_close(self, data: dict):
        """Close episode."""
        episode_id = data.get("episode_id")
        self.manager.remove(episode_id)
        self._send_json({"status": "closed", "episode_id": episode_id})


def main():
    parser = argparse.ArgumentParser(description="OSWorld HTTP Server")
    parser.add_argument("--port", type=int, default=8100, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--max-episodes", type=int, default=10, help="Max concurrent episodes")
    args = parser.parse_args()

    # Verify OSWorld is available
    try:
        get_desktop_env_class()
        logger.info("OSWorld desktop_env loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import desktop_env: {e}")
        logger.error("Make sure you're running in the osworld_venv with desktop-env installed")
        return 1

    # Setup handler with episode manager
    OSWorldHandler.manager = EpisodeManager(max_episodes=args.max_episodes)

    server = HTTPServer((args.host, args.port), OSWorldHandler)
    logger.info(f"OSWorld server starting on {args.host}:{args.port}")
    logger.info(f"Container should use: OSWORLD_SERVER_URL=http://172.17.0.1:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()

    return 0


if __name__ == "__main__":
    exit(main())
