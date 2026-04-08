from __future__ import annotations

from ..models import StateModel, TaskGrade, TaskSpec
from .common import grade_single_ticket


WEIGHTS = {
    "context": 0.25,
    "priority": 0.15,
    "route": 0.25,
    "resolution": 0.25,
    "escalation": 0.1,
}


def grade(task: TaskSpec, state: StateModel) -> TaskGrade:
    return grade_single_ticket(task, state, WEIGHTS)
