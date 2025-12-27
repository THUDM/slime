from __future__ import annotations

import os
import random
import re
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from PIL import Image, ImageDraw
from slime.utils.types import Sample


@dataclass
class SokobanEnvConfig:
    grid_size: tuple[int, int] = (6, 6)
    num_boxes: int = 1
    max_steps: int = 100
    render_mode: str = "vision"  # "vision" or "text"
    text_grid_in_vision: bool = False  # whether to include the ASCII grid alongside the image
    initial_base_grid: list[list[int]] | None = None
    initial_boxes: list[list[int]] | None = None
    initial_player: list[int] | None = None
    max_actions_per_step: int = 3
    action_sep: str = ","
    image_placeholder: str = "<image>"
    rng_seed: int | None = None
    wall_fraction: float = 0.05  # chance to place an internal wall
    log_dir: str | None = None
    log_prefix: str = "step"

    def copy_with_updates(self, overrides: dict | None) -> "SokobanEnvConfig":
        return SokobanEnvConfig(**{**self.__dict__, **(overrides or {})})


class SokobanEnv:
    SYMBOLS = {
        0: "#",  # wall
        1: "_",  # floor
        2: "O",  # target
        3: "√",  # box on target
        4: "X",  # box
        5: "P",  # player
        6: "S",  # player on target
    }

    ACTION_DELTAS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(self, config: SokobanEnvConfig | None = None):
        self.config = config or SokobanEnvConfig()
        self.rng = random.Random(self.config.rng_seed)
        self.base_grid: list[list[int]] = []
        self.boxes: set[tuple[int, int]] = set()
        self.player: tuple[int, int] | None = None
        self.steps = 0
        self.render_count = 0
        self.step_rewards: list[float] = []

    # ----------------- Helpers ----------------- #
    def _empty_base_grid(self) -> list[list[int]]:
        h, w = self.config.grid_size
        grid = [[1 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if r in {0, h - 1} or c in {0, w - 1}:
                    grid[r][c] = 0  # border walls
                elif self.rng.random() < self.config.wall_fraction:
                    grid[r][c] = 0
        return grid

    def _random_empty_cell(self, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        h, w = self.config.grid_size
        attempts = 0
        while True:
            r = self.rng.randint(1, h - 2)
            c = self.rng.randint(1, w - 2)
            if (r, c) in occupied or self.base_grid[r][c] == 0:
                attempts += 1
                if attempts > 1000:
                    raise RuntimeError("Failed to sample empty cell for Sokoban.")
                continue
            return (r, c)

    def _compose_cell_value(self, r: int, c: int) -> int:
        if (r, c) == self.player:
            return 6 if self.base_grid[r][c] == 2 else 5
        if (r, c) in self.boxes:
            return 3 if self.base_grid[r][c] == 2 else 4
        return self.base_grid[r][c]

    def _render_text(self) -> str:
        rows = []
        h, w = self.config.grid_size
        for r in range(h):
            row_chars = []
            for c in range(w):
                val = self._compose_cell_value(r, c)
                row_chars.append(f" {self.SYMBOLS[val]} ")
            rows.append("\t".join(row_chars))
        return "\n".join(rows)

    @staticmethod
    def _manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def box_target_distance(self) -> float:
        """
        Compute a simple distance metric between boxes and targets.
        For the single-box case this is the Manhattan distance to the target.
        For multiple boxes, we greedily assign each box to its nearest target.
        """
        targets = [(r, c) for r, row in enumerate(self.base_grid) for c, val in enumerate(row) if val == 2]
        if not targets or not self.boxes:
            return 0.0

        remaining_targets = set(targets)
        total_distance = 0.0

        for box in sorted(self.boxes):
            best_target = None
            best_dist = None
            for target in remaining_targets:
                dist = self._manhattan_distance(box, target)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_target = target
            if best_dist is None:
                continue
            total_distance += float(best_dist)
            if best_target in remaining_targets and len(remaining_targets) > 1:
                remaining_targets.remove(best_target)

        return total_distance

    def snapshot_text_state(self) -> str:
        """Expose a textual rendering of the current grid for logging/rewards."""
        return self._render_text()

    def _render_image(self, cell_size: int = 64) -> Image.Image:
        """
        Render a Sokoban grid in the same style as the reference image:
        - Red brick walls with light mortar
        - Black floor
        - Red outlined targets with a red diamond
        - Yellow box with orange X
        - Green alien-shaped player
        """
        h, w = self.config.grid_size
        img = Image.new("RGB", (w * cell_size, h * cell_size), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        wall_base = (176, 80, 50)  # warm red brick
        wall_lines = (232, 200, 180)  # light mortar lines
        floor_color = (8, 8, 8)  # near-black floor

        def draw_brick_cell(x0: int, y0: int, x1: int, y1: int):
            # Solid brick color only (no visible pattern)
            draw.rectangle([x0, y0, x1, y1], fill=wall_base)

        def draw_floor_cell(x0: int, y0: int, x1: int, y1: int):
            draw.rectangle([x0, y0, x1, y1], fill=floor_color)

        def draw_target(x0: int, y0: int, x1: int, y1: int):
            draw_floor_cell(x0, y0, x1, y1)
            pad = (x1 - x0) // 4
            draw.rectangle([x0 + pad, y0 + pad, x1 - pad, y1 - pad], outline=(220, 40, 40), width=3)
            # red diamond
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            diamond_r = (x1 - x0) // 6
            draw.polygon(
                [(cx, cy - diamond_r), (cx + diamond_r, cy), (cx, cy + diamond_r), (cx - diamond_r, cy)],
                fill=(200, 30, 30),
            )

        def draw_box(x0: int, y0: int, x1: int, y1: int):
            draw.rectangle([x0, y0, x1, y1], fill=(233, 184, 47), outline=(180, 130, 20), width=3)
            draw.line([x0 + 6, y0 + 6, x1 - 6, y1 - 6], fill=(196, 140, 24), width=4)
            draw.line([x0 + 6, y1 - 6, x1 - 6, y0 + 6], fill=(196, 140, 24), width=4)

        def draw_player(x0: int, y0: int, x1: int, y1: int):
            # alien-like silhouette
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            body_r = (x1 - x0) // 4
            head_r = body_r // 2
            color = (70, 200, 70)

            # head with small antennae
            draw.ellipse([cx - head_r, y0, cx + head_r, y0 + head_r * 2], fill=color)
            antenna_w = max(2, head_r // 3)
            draw.line([cx - head_r // 2, y0 - antenna_w, cx - head_r // 2, y0], fill=color, width=antenna_w)
            draw.line([cx + head_r // 2, y0 - antenna_w, cx + head_r // 2, y0], fill=color, width=antenna_w)

            # body
            draw.rectangle([cx - body_r, y0 + head_r * 2 - 2, cx + body_r, cy + body_r], fill=color)

            # eyes
            eye_r = max(2, head_r // 3)
            draw.ellipse([cx - head_r // 2 - eye_r, y0 + head_r - eye_r, cx - head_r // 2 + eye_r, y0 + head_r + eye_r], fill=(0, 0, 0))
            draw.ellipse([cx + head_r // 2 - eye_r, y0 + head_r - eye_r, cx + head_r // 2 + eye_r, y0 + head_r + eye_r], fill=(0, 0, 0))

            # legs
            leg_w = max(2, body_r // 3)
            draw.rectangle([cx - body_r, cy + body_r, cx - body_r + leg_w, y1], fill=color)
            draw.rectangle([cx + body_r - leg_w, cy + body_r, cx + body_r, y1], fill=color)

        grid_color = (255, 255, 255)
        for r in range(h):
            for c in range(w):
                val = self._compose_cell_value(r, c)
                x0, y0 = c * cell_size, r * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size

                base = self.base_grid[r][c]
                if base == 0:
                    draw_brick_cell(x0, y0, x1, y1)
                elif base == 2:
                    draw_target(x0, y0, x1, y1)
                    draw.line([x0, y0, x1, y0], fill=grid_color, width=1)
                    draw.line([x0, y1, x1, y1], fill=grid_color, width=1)
                    draw.line([x0, y0, x0, y1], fill=grid_color, width=1)
                    draw.line([x1, y0, x1, y1], fill=grid_color, width=1)
                else:
                    draw_floor_cell(x0, y0, x1, y1)
                    draw.line([x0, y0, x1, y0], fill=grid_color, width=1)
                    draw.line([x0, y1, x1, y1], fill=grid_color, width=1)
                    draw.line([x0, y0, x0, y1], fill=grid_color, width=1)
                    draw.line([x1, y0, x1, y1], fill=grid_color, width=1)

                # overlay objects
                if val in (3, 4):  # box
                    draw_box(x0 + 4, y0 + 4, x1 - 4, y1 - 4)
                if val in (5, 6):  # player
                    draw_player(x0 + 6, y0 + 6, x1 - 6, y1 - 6)

        return img

    def _obs_template(self, observation: str, valid_actions: Iterable[str]) -> str:
        actions = ", ".join(valid_actions) if valid_actions else "None"
        return (
            f"[Observation]\n{observation}\n"
            f"Valid last actions: {actions}\n"
            f"Provide your next move(s) using Up, Down, Left, Right."
        )

    def _parse_actions(self, action_str: str) -> list[str]:
        # Try to extract content inside <answer> tags first
        m = re.search(r"<answer>(.*?)</answer>", action_str, flags=re.IGNORECASE | re.DOTALL)
        candidate = m.group(1) if m else action_str
        splits = re.split(r"[,;\n]+|\s+", candidate)
        actions = []
        for token in splits:
            t = token.strip().lower()
            if t in self.ACTION_DELTAS:
                actions.append(t)
            if len(actions) >= self.config.max_actions_per_step:
                break
        return actions

    # ----------------- Public API ----------------- #
    def reset(self):
        self.steps = 0
        self.step_rewards = []

        # Deterministic reset from provided initial state (used when loading dataset states)
        init_base = self.config.initial_base_grid
        init_boxes = self.config.initial_boxes
        init_player = self.config.initial_player
        if init_base is not None and init_boxes is not None and init_player is not None:
            base_grid = init_base.tolist() if hasattr(init_base, "tolist") else init_base
            boxes_list = init_boxes.tolist() if hasattr(init_boxes, "tolist") else init_boxes
            player_list = init_player.tolist() if hasattr(init_player, "tolist") else init_player

            # Normalize to Python primitives
            self.base_grid = deepcopy(base_grid)
            self.boxes = {tuple(b) for b in boxes_list}
            self.player = tuple(player_list)
            # Ensure config matches provided initial state dimensions/counts
            self.config.grid_size = (len(self.base_grid), len(self.base_grid[0]) if self.base_grid else 0)
            self.config.num_boxes = len(self.boxes)
            obs = self._render_observation(init_obs=True, valid_actions=[])
            return obs, {}

        # regenerate until solvable (only enforced for single-box case)
        for attempt in range(100):
            self.base_grid = self._empty_base_grid()
            self.boxes = set()
            occupied: set[tuple[int, int]] = set()

            # targets
            targets = []
            for _ in range(self.config.num_boxes):
                pos = self._random_empty_cell(occupied)
                targets.append(pos)
                occupied.add(pos)
                self.base_grid[pos[0]][pos[1]] = 2

            # boxes
            for _ in range(self.config.num_boxes):
                pos = self._random_empty_cell(occupied)
                self.boxes.add(pos)
                occupied.add(pos)

            # player
            self.player = self._random_empty_cell(occupied)

            if self.config.num_boxes == 1:
                if self._is_solvable_single_box(targets[0]):
                    break
            else:
                break
        else:
            # fallback to a simple solvable layout
            h, w = self.config.grid_size
            self.base_grid = [[0 if r in {0, h - 1} or c in {0, w - 1} else 1 for c in range(w)] for r in range(h)]
            target = (h - 2, w - 2)
            self.base_grid[target[0]][target[1]] = 2
            self.boxes = {(1, 1)}
            self.player = (1, 2)

        obs = self._render_observation(init_obs=True, valid_actions=[])
        return obs, {}

    def _render_observation(self, init_obs: bool, valid_actions: list[str]) -> dict:
        include_text_grid = self.config.render_mode != "vision" or self.config.text_grid_in_vision
        obs_text = self._render_text() if include_text_grid else "(Refer to the image for the grid layout.)"
        if self.config.render_mode == "vision":
            image = self._render_image()
            payload = {
                "obs_str": self._obs_template(obs_text, valid_actions),
                "multi_modal_data": {self.config.image_placeholder: [image]},
            }

            if self.config.log_dir:
                os.makedirs(self.config.log_dir, exist_ok=True)
                image_path = os.path.join(
                    self.config.log_dir, f"{self.config.log_prefix}_{self.render_count}.png"
                )
                try:
                    image.save(image_path)
                except Exception:
                    pass
                self.render_count += 1
        else:
            payload = {
                "obs_str": self._obs_template(obs_text, valid_actions),
            }
        return payload

    def _move(self, action: str) -> tuple[bool, float]:
        """Execute a single primitive move. Returns (effective_move, reward_delta)."""
        dr, dc = self.ACTION_DELTAS[action]
        pr, pc = self.player
        nr, nc = pr + dr, pc + dc
        h, w = self.config.grid_size
        reward = 0.0

        # Bounds / wall check
        if nr < 0 or nr >= h or nc < 0 or nc >= w or self.base_grid[nr][nc] == 0:
            return False, -0.05

        # Box push
        if (nr, nc) in self.boxes:
            br, bc = nr + dr, nc + dc
            if br < 0 or br >= h or bc < 0 or bc >= w or self.base_grid[br][bc] == 0 or (br, bc) in self.boxes:
                return False, -0.05
            # move box
            self.boxes.remove((nr, nc))
            self.boxes.add((br, bc))
            if self.base_grid[br][bc] == 2:
                reward += 1.0  # box onto target
            if self.base_grid[nr][nc] == 2:
                reward -= 0.5  # box left target

        # move player
        self.player = (nr, nc)
        reward += 0.01  # small step bonus for effective move
        return True, reward

    #called during rollout after receive response from model 
    def step(self, action_str: str):
        actions = self._parse_actions(action_str)
        action_is_valid = len(actions) > 0
        effective = False
        reward = 0.0
        info = {
            "raw_action": action_str,
            "parsed_actions": actions,
            "metrics": {
                "turn_metrics": {
                    "action_is_valid": action_is_valid,
                    "action_is_effective": False,
                },
                "traj_metrics": {"success": False},
            },
        }

        if not actions:
            obs = self._render_observation(init_obs=False, valid_actions=[])
            done = False
            self.step_rewards.append(reward)
            return obs, done, info

        for act in actions:
            moved, delta = self._move(act)
            reward += delta
            effective = effective or moved
            self.steps += 1
            if self._success():
                info["metrics"]["traj_metrics"]["success"] = True
                break
            if self.steps >= self.config.max_steps:
                break

        info["metrics"]["turn_metrics"]["action_is_effective"] = effective

        done = info["metrics"]["traj_metrics"]["success"] or self.steps >= self.config.max_steps
        self.step_rewards.append(reward)
        obs = self._render_observation(init_obs=False, valid_actions=actions)
        return obs, done, info

    def _success(self) -> bool:
        return all(self.base_grid[r][c] == 2 for (r, c) in self.boxes)

    def _is_solvable_single_box(self, target: tuple[int, int]) -> bool:
        """Simple BFS on (player, box) state to ensure solvable layout for single box."""
        if not self.boxes:
            return False
        start_box = next(iter(self.boxes))
        start_player = self.player
        walls = {(r, c) for r, row in enumerate(self.base_grid) for c, v in enumerate(row) if v == 0}
        h, w = self.config.grid_size

        def neighbors(state):
            p_r, p_c, b_r, b_c = state
            for dr, dc in self.ACTION_DELTAS.values():
                np_r, np_c = p_r + dr, p_c + dc
                if np_r < 0 or np_r >= h or np_c < 0 or np_c >= w or (np_r, np_c) in walls:
                    continue
                nb_r, nb_c = b_r, b_c
                # pushing?
                if (np_r, np_c) == (b_r, b_c):
                    push_r, push_c = b_r + dr, b_c + dc
                    if (
                        push_r < 0
                        or push_r >= h
                        or push_c < 0
                        or push_c >= w
                        or (push_r, push_c) in walls
                        or (push_r, push_c) in self.boxes
                    ):
                        continue
                    nb_r, nb_c = push_r, push_c
                yield (np_r, np_c, nb_r, nb_c)

        start = (start_player[0], start_player[1], start_box[0], start_box[1])
        queue = deque([start])
        visited = {start}

        while queue:
            p_r, p_c, b_r, b_c = queue.popleft()
            if (b_r, b_c) == target:
                return True
            for nxt in neighbors((p_r, p_c, b_r, b_c)):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def system_prompt(self) -> str:
        symbols = "# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target"
        return (
            "You are a Sokoban solver.\n"
            f"Grid legend: {symbols}\n"
            "Goal: push all boxes onto targets. You can move Up, Down, Left, Right.\n"
            "Rules:\n"
            "1) Push boxes; you cannot pull. 2) Avoid walls. 3) Do not move a box off a target.\n"
            "4) Do not repeat an invalid or ineffective action—choose a different direction if the last move failed.\n"
            "You must always produce BOTH thinking and action blocks.\n"
            "<think>\n"
            "- Briefly describe where you are relative to box(es) and target(s).\n"
            "- State a short plan to move a box onto its target.\n"
            "</think>\n"
            "<answer>...</answer>  # Action must be exactly one of Up/Down/Left/Right (no filler)"
        )

    def compute_reward(self) -> float:
        # Final bonus for success
        return 5.0 if self._success() else 0.0

    def close(self):
        return None


# ----------------- Rollout Integration Helpers ----------------- #
DEFAULT_ROLLOUT_CONFIG: dict[str, Any] = {
    "max_turns": 20,
    "max_total_tokens": 8192,
    "stop_on_max_tokens": True,
}

# Base environment defaults; can be overridden via sample.metadata["env_config"].
DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "render_mode": "vision",
    "max_actions_per_step": 3,
    "grid_size": (6, 6),
    "num_boxes": 1,
    "max_steps": 100,
}


def build_env(sample: Sample | None = None, args: Any | None = None, config_overrides: dict | None = None) -> SokobanEnv:
    """
    Construct a Sokoban environment using defaults plus any per-sample overrides.

    Args:
        sample: rollout sample whose metadata may include an 'env_config' dict.
        args: unused placeholder for signature consistency with other env modules.
        config_overrides: optional config overrides applied after defaults.
    """
    env_kwargs = deepcopy(DEFAULT_ENV_CONFIG)
    if config_overrides:
        env_kwargs.update(deepcopy(config_overrides))

    sample_metadata = getattr(sample, "metadata", None) or {}
    env_kwargs.update(deepcopy(sample_metadata.get("env_config", {})))

    env_cfg = SokobanEnvConfig(**env_kwargs)
    return SokobanEnv(env_cfg)


def format_observation(observation: dict) -> dict:
    """Convert an observation payload into a chat message with optional images."""
    content = []
    multimodal = observation.get("multi_modal_data") or {}
    for _, images in multimodal.items():
        for image in images:
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": observation.get("obs_str", "")})
    return {"role": "user", "content": content}


def on_reset(env: SokobanEnv, observation: dict, sample: Sample | None = None, reset_info: dict | None = None) -> dict:
    """
    Capture environment metadata immediately after reset so rollout can persist it.
    """
    metadata = getattr(sample, "metadata", None) or {}
    env_meta = deepcopy(metadata.get("sokoban", {}))
    env_meta.update(
        {
            "seed": metadata.get("seed"),
            "initial_state_text": env.snapshot_text_state(),
            "initial_obs_text": observation.get("obs_str", "") if isinstance(observation, dict) else "",
            "dist0": env.box_target_distance(),
        }
    )
    return {"sokoban": env_meta}


def finalize_episode(
    env: SokobanEnv,
    observation: dict,
    sample: Sample | None = None,
    responses: list[str] | None = None,
) -> dict:
    """
    Collect trajectory metadata; reward is computed in the custom reward function.
    """
    metadata = getattr(sample, "metadata", None) or {}
    env_meta = deepcopy(metadata.get("sokoban", {}))

    dist0 = env_meta.get("dist0")
    dist1 = env.box_target_distance()
    final_bonus = env.compute_reward()

    env_meta.update(
        {
            "turns": len(responses or []),
            "final_state_text": env.snapshot_text_state(),
            "final_obs_text": observation.get("obs_str", "") if isinstance(observation, dict) else "",
            "dist1": dist1,
            "turn_rewards": list(env.step_rewards),
            "final_bonus": final_bonus,
        }
    )
    return {"sokoban": env_meta}
