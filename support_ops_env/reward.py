from __future__ import annotations

from typing import Dict

from .models import RewardModel


STEP_PENALTY = -0.01


def build_reward(components: Dict[str, float], rationale: str) -> RewardModel:
    value = round(sum(components.values()), 4)
    return RewardModel(value=value, components=components, rationale=rationale)
