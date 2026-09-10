from __future__ import annotations

import json
import logging
import re
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None
from examples.opencua_vlm_multi_turn.base_env import BaseInteractionEnv

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Matches the JSON payload emitted between <tool_call> ... </tool_call> tags.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Accept either name; verl uses `calc_opencua_reward` while the instruction refers to `calc_score`.
SUPPORTED_TOOL_NAMES = {"calc_score", "calc_opencua_reward"}


class OpencuaEnv(BaseInteractionEnv):
    """
    Minimal environment for OpenCUA tasks where the model outputs a \boxed{(x,y)} response.

    Reward is 1.0 if the predicted point is inside the ground-truth relative_bbox,
    0.0 otherwise. Feedback is provided in English indicating if the point is
    left/right/up/down relative to the target bbox.
    """

    def __init__(self, *, ground_truth: str | None = None, max_turns: int | None = None):
        """
        Args:
            ground_truth (str): JSON string of relative bbox [x_min, y_min, x_max, y_max]
            max_turns (int | None): maximum allowed steps per episode
        """
        self.ground_truth = str(ground_truth) if ground_truth is not None else None
        self.last_score: float | None = None
        self.turn = 0
        self.max_turns = max_turns

    def reset(self):
        """Reset the environment to start a new episode."""
        self.turn = 0
        self.last_score = None
        observation = {}  # No initial observation needed
        reset_info = {"ground_truth_available": self.ground_truth is not None}
        return observation, reset_info

    def close(self):
        """No resources to release in this minimal environment."""
        return

    # -----------------------------
    # Reward computation
    # -----------------------------
    def _score_answer(self, answer: str) -> float:
        """
        Parse the answer string \boxed{(x,y)} and check if the point lies
        inside the ground-truth bbox.
        """
        if not self.ground_truth:
            return 0.0
        try:
            bbox = json.loads(self.ground_truth)
            x_min, y_min, x_max, y_max = bbox

        except Exception:
            return 0.0

        # Parse the \boxed{(x,y)} pattern
        m = re.search(r"\\boxed\s*\{\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*\}", answer)
        logger.info(f"Parse the boxed pattern, m={m},answer={answer}")
        if not m:
            return 0.0
        try:
            x = float(m.group(1))
            y = float(m.group(2))
            logger.info(f"score_answer,boxed x={x},y={y}")
        except Exception:
            logger.info("m can not be extracted")
            return 0.0

        # Reward = 1 if point is inside bbox
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return 1.0
        elif x > 1 or y > 1 or x < 0 or y < 0:
            return 0.0  # not relative grounding
        else:
            return 0.2  # realtive grounding, format reward

    # -----------------------------
    # Directional feedback
    # -----------------------------
    def _directional_feedback(self, x: float, y: float, bbox: list[float]) -> str:
        """
        Provide English feedback about the predicted point relative to bbox.

        Returns a string like:
            "Point is outside the box: slightly left and slightly above."
            "Point is inside the box."
        """
        x_min, y_min, x_max, y_max = bbox
        horz, vert = "", ""

        # Horizontal feedback
        if x < x_min:
            horz = "slightly left"
        elif x > x_max:
            horz = "slightly right"
        else:
            horz = "horizontally aligned"

        # Vertical feedback
        if y < y_min:
            vert = "slightly above"
        elif y > y_max:
            vert = "slightly below"
        else:
            vert = "vertically aligned"

        if horz == "horizontally aligned" and vert == "vertically aligned":
            return "Point is inside the box."
        if x < 0 or x > 1 or y < 0 or y > 1:
            return "The range of x and y must be in [0,1]"
        return f"Point is outside the box: {horz} and {vert}."

    # -----------------------------
    # Step function
    # -----------------------------
    def step(self, response_text: str):
        """
        Step function for the environment.

        Args:
            response_text (str): model response, must contain \boxed{(x,y)}

        Returns:
            obs (dict): observation dictionary with 'obs_str' and 'tool_score'
            done (bool): whether the episode is finished
            info (dict): additional information including parsed_answer and score
        """
        self.turn += 1
        done = self.max_turns is not None and self.turn >= self.max_turns

        parsed_answer = response_text.strip()
        score = self._score_answer(parsed_answer)
        logger.info(f"parsed_answer={parsed_answer},score={score}")
        self.last_score = score

        # Parse the predicted point for feedback
        try:
            bbox = json.loads(self.ground_truth)
            m = re.search(r"\\boxed\(\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)\s*\)", parsed_answer)
            if m:
                x_pred = float(m.group(1))
                y_pred = float(m.group(2))
                direction_feedback = self._directional_feedback(x_pred, y_pred, bbox)
            else:
                direction_feedback = "No valid point found in the answer."
        except Exception:
            direction_feedback = "Failed to parse ground-truth bbox."

        obs_str = f"{direction_feedback} (score={score})"
        obs = {"obs_str": obs_str, "role": "tool", "tool_score": score}

        info = {
            "parsed_answer": parsed_answer,
            "score": score,
            "turn": self.turn,
            "tool_executed": True,
            "answer_missing": True if score == 0.0 else False,
        }

        return obs, done, info


def _extract_ground_truth(sample: Sample | None) -> str | None:
    """Resolve the ground-truth answer from label or metadata."""
    if sample is None:
        return None
    if sample.label is not None:
        return str(sample.label)
    # metadata = sample.metadata
    # for key in ("answer", "ground_truth", "label"):
    #     if key in metadata and metadata[key] is not None:
    #         return str(metadata[key])
    return None


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> OpencuaEnv:
    """
    Construct a OpencuaEnv. Ground truth is pulled from sample.label or metadata.
    """
    ground_truth = _extract_ground_truth(sample)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    if ground_truth is None:
        logger.warning("Ground truth answer missing; calc_score tool will always return 0.")
    return OpencuaEnv(ground_truth=ground_truth, max_turns=max_turns)
