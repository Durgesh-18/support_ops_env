from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ..models import TaskSpec


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_file(name: str) -> List[TaskSpec]:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [TaskSpec.model_validate(item) for item in raw]


def get_all_tasks() -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    for filename in ("easy_cases.json", "medium_cases.json", "hard_cases.json"):
        tasks.extend(_load_file(filename))
    return tasks


def get_task(task_id: str) -> TaskSpec:
    for task in get_all_tasks():
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown task_id: {task_id}")


def list_task_ids() -> List[str]:
    return [task.task_id for task in get_all_tasks()]
