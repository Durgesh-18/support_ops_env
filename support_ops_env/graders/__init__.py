from __future__ import annotations

from typing import Callable, Dict

from ..models import StateModel, TaskGrade, TaskSpec
from .easy import grade as easy_grade
from .hard import grade as hard_grade
from .medium import grade as medium_grade


GRADERS: Dict[str, Callable[[TaskSpec, StateModel], TaskGrade]] = {
    "easy_support_routing": easy_grade,
    "medium_support_resolution": medium_grade,
    "hard_support_queue": hard_grade,
}


def grade_task(task: TaskSpec, state: StateModel) -> TaskGrade:
    return GRADERS[task.grader_name](task, state)
