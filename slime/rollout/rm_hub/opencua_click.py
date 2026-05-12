import json
import logging
import math
import re

logger = logging.getLogger(__name__)


def grounding_reward(x, y, x_min, y_min, x_max, y_max, alpha=10.0):
    # 1. hard constraint
    if x < 0 or x > 1 or y < 0 or y > 1:
        return -0.5

    # 2. inside box
    if x_min <= x <= x_max and y_min <= y <= y_max:
        return 1.0

    # 3. soft L-infinity distance
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)

    d_soft = math.log(math.exp(alpha * dx) + math.exp(alpha * dy)) / alpha

    # 4. map distance to reward
    reward = math.exp(-5.0 * d_soft)  # or 1 - d_soft

    return max(reward, 0.0)


def compute_opencua_reward(solution_str: str, ground_truth: str, metadata):
    if not ground_truth:
        return 0.0
    try:
        bbox = json.loads(ground_truth)
        x_min, y_min, x_max, y_max = bbox

    except Exception:
        logger.info(f"there is no ground truth,ground_truth={ground_truth}")
        return 0.0
    if not solution_str:
        logger.info("there is no solution.")
        return 0.0
    m = re.search(r"\\boxed\s*\{\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*\}", solution_str)
    logger.info(f"Parse the boxed pattern, m={m},solution_str={solution_str}")
    if not m:
        logger.info("Can not extract valid output")
        return -1.0
    try:
        x = float(m.group(1))
        y = float(m.group(2))
        logger.info(f"score_solution_str,boxed x={x},y={y}")
    except Exception:
        logger.info("m can not be extracted")
        return -1.0
    # Smooth Chebyshev
    return grounding_reward(x, y, x_min, y_min, x_max, y_max)
