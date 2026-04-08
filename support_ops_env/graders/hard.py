from __future__ import annotations

from ..models import StateModel, TaskGrade, TaskSpec
from .common import grade_queue_task


WEIGHTS = {
    "context": 0.1,
    "priority": 0.2,
    "route": 0.25,
    "resolution": 0.2,
    "escalation": 0.1,
    "ranking": 0.15,
}


def grade(task: TaskSpec, state: StateModel) -> TaskGrade:
    return grade_queue_task(task, state, WEIGHTS)
