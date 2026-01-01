"""RLVE Prompt Provider - on-the-fly environment sampling and prompt generation."""

import json
import logging
import os
import random
from collections import defaultdict
from typing import Any

import yaml

logger = logging.getLogger(__name__)

try:
    from Gym.environments import identifier2environment

    RLVE_AVAILABLE = True
except ImportError:
    RLVE_AVAILABLE = False
    identifier2environment = {}
    logger.warning(
        "RLVE Gym not installed. Using stub mode (CI only). "
        "Install: pip install rlve-gym"
    )

try:
    from Gym.parameter_controllers import identifier2controller
except ImportError:
    identifier2controller = {}


class RLVEPromptProvider:
    """Samples environments by weight, generates fresh problems each rollout."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.env_ids: list[str] = []
        self.weights: list[float] = []
        for env_id, env_cfg in self.config["environments"].items():
            self.env_ids.append(env_id)
            self.weights.append(env_cfg.get("weight", 1.0))

        if RLVE_AVAILABLE:
            for env_id in self.env_ids:
                if env_id not in identifier2environment:
                    raise ValueError(
                        f"Environment '{env_id}' not found. "
                        f"Available: {list(identifier2environment.keys())[:10]}..."
                    )

        markers = self.config.get("answer_markers", {})
        self.answer_start = markers.get("start", "<answer>")
        self.answer_end = markers.get("end", "</answer>")

        # RLVE-style difficulty tracking per environment
        initial_difficulty = self.config.get("initial_difficulty", 0)
        self.environment2difficulty: dict[str, int] = {
            env_id: initial_difficulty for env_id in self.env_ids
        }
        self.difficulty_sliding_window_size = self.config.get(
            "difficulty_sliding_window_size", 1
        )

        self.env_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"acc": 0.0, "count": 0}
        )
        self.min_prompts_before_difficulty_check = self.config.get(
            "min_prompts_before_difficulty_check",
            8,
        )
        self.min_metric_to_increase_difficulty = self.config.get(
            "min_metric_to_increase_difficulty",
            0.5,
        )

        # Problem generation seed for reproducibility (matches RLVE's RLVEManager)
        self.problem_generation_seed = 0

    def get_prompt(self) -> dict[str, Any]:
        """Sample environment and generate problem. Returns prompt and metadata."""
        if not RLVE_AVAILABLE:
            return self._get_stub_prompt()

        env_id = random.choices(self.env_ids, weights=self.weights, k=1)[0]
        env_class = identifier2environment[env_id]

        problem_difficulty: int | None = None
        maximum_difficulty: int | None = None

        if self.config.get("use_controllers", False) and env_id in identifier2controller:
            controller_cls = identifier2controller[env_id]
            controller = controller_cls()

            maximum_difficulty = self.environment2difficulty.get(env_id, 0)
            window = self.difficulty_sliding_window_size

            difficulty_buckets: list[tuple[int, list[dict[str, Any]]]] = []
            for d in range(maximum_difficulty + 1):
                if d > maximum_difficulty - window:
                    params = list(controller.get_parameter_list())
                    difficulty_buckets.append((d, params))
                controller.update()

            if difficulty_buckets:
                problem_difficulty, param_list = random.choice(difficulty_buckets)
                parameter = random.choice(param_list)
            else:
                parameter = self.config["environments"][env_id].get("kwargs", {})
        else:
            parameter = self.config["environments"][env_id].get("kwargs", {})

        env = env_class(answer_markers=(self.answer_start, self.answer_end))
        env.generator(seed=self.problem_generation_seed, parameter=parameter)
        self.problem_generation_seed += 1

        prompt = env.prompt_generator()
        prompt += (
            f"\n\nWrap your final answer with "
            f"{self.answer_start}...{self.answer_end} tags."
        )

        return {
            "prompt": prompt,
            "metadata": {
                "env_id": env_id,
                "config": env.get_config(),
                "answer_markers": (self.answer_start, self.answer_end),
                "difficulty_index": problem_difficulty,
                "max_difficulty_index": maximum_difficulty,
            },
        }

    def _get_stub_prompt(self) -> dict[str, Any]:
        """Stub prompt for CI testing."""
        random.seed(self.problem_generation_seed)
        self.problem_generation_seed += 1
        a, b = random.randint(1, 100), random.randint(1, 100)
        prompt = f"What is {a} + {b}?\n\nWrap your final answer with <answer>...</answer> tags."
        return {
            "prompt": prompt,
            "metadata": {
                "env_id": "StubAddition",
                "config": {"a": a, "b": b, "answer": a + b},
            },
        }

    def update_controller(
        self,
        env_id: str,
        accuracy: float,
        difficulty_index: int | None,
        n_samples_per_prompt: int,
    ) -> None:
        """Track accuracy and advance difficulty when threshold exceeded."""
        if not self.config.get("use_controllers", False):
            return

        if env_id not in self.environment2difficulty:
            return

        maximum_difficulty = self.environment2difficulty[env_id]

        if difficulty_index is None:
            return
        if difficulty_index < maximum_difficulty:
            return

        stats = self.env_stats[env_id]
        stats["acc"] += accuracy
        stats["count"] += 1

        threshold = (
            self.min_prompts_before_difficulty_check * max(n_samples_per_prompt, 1)
        )
        if stats["count"] >= threshold:
            avg_acc = stats["acc"] / stats["count"]
            if avg_acc >= self.min_metric_to_increase_difficulty:
                self.environment2difficulty[env_id] = maximum_difficulty + 1
                logger.info(
                    f"[RLVE] {env_id}: difficulty "
                    f"{maximum_difficulty} -> {maximum_difficulty + 1} "
                    f"(avg_acc={avg_acc:.2f})"
                )
            stats["acc"] = 0.0
            stats["count"] = 0

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for checkpointing (matches RLVE's RLVEManager.get_state)."""
        return {
            "environment2difficulty": dict(self.environment2difficulty),
            "environment2accuracy": {
                env_id: {"accuracy": stats["acc"], "num_samples": int(stats["count"])}
                for env_id, stats in self.env_stats.items()
            },
            "problem_generation_seed": self.problem_generation_seed,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint (matches RLVE's RLVEManager.set_state)."""
        self.environment2difficulty = state["environment2difficulty"]
        for env_id, acc_state in state.get("environment2accuracy", {}).items():
            self.env_stats[env_id]["acc"] = acc_state.get("accuracy", 0.0)
            self.env_stats[env_id]["count"] = acc_state.get("num_samples", 0)
        self.problem_generation_seed = state.get("problem_generation_seed", 0)

    def save_state(self, path: str) -> str:
        """Save state to JSON file for checkpoint/resume."""
        state = self.get_state()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"[RLVE] Saved state to {path}")
        return path

    def load_state(self, path: str) -> None:
        """Load state from JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"State file not found: {path}")
        with open(path) as f:
            state = json.load(f)
        self.set_state(state)
        logger.info(f"[RLVE] Loaded state from {path}")


def _get_config_path() -> str:
    default_path = os.path.join(os.path.dirname(__file__), "configs", "starter_pack.yaml")
    return os.environ.get("RLVE_CONFIG_PATH", default_path)


_provider_singleton: RLVEPromptProvider | None = None


def get_provider() -> RLVEPromptProvider:
    """Get or create the global provider singleton."""
    global _provider_singleton
    if _provider_singleton is None:
        config_path = _get_config_path()
        logger.info(f"Initializing RLVEPromptProvider from {config_path}")
        _provider_singleton = RLVEPromptProvider(config_path)
    return _provider_singleton
