"""OpenEnv task validation: every emitted score must satisfy 0 < x < 1 (never 0.0 or 1.0)."""

from __future__ import annotations

MIN_REPORTED_REWARD = 0.001
MAX_REPORTED_REWARD = 0.85  # was 0.9 — avoid hitting ceiling exactly
_INTERIOR_CEILING = 0.9999


def clamp_open_interval(value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return MIN_REPORTED_REWARD
    rounded = round(x, 4)
    if rounded <= 0.0:
        return MIN_REPORTED_REWARD
    if rounded >= 1.0:
        return MAX_REPORTED_REWARD
    bounded = min(max(rounded, MIN_REPORTED_REWARD), _INTERIOR_CEILING)
    return round(bounded, 4)
