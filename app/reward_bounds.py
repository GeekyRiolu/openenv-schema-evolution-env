"""OpenEnv task validation requires every reported score to lie strictly in (0, 1)."""

from __future__ import annotations

MIN_REPORTED_REWARD = 0.001
MAX_REPORTED_REWARD = 0.9


def clamp_open_interval(value: float) -> float:
    rounded = round(value, 4)
    if rounded <= 0.0:
        return MIN_REPORTED_REWARD
    if rounded >= 1.0:
        return MAX_REPORTED_REWARD
    return rounded
