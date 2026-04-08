from __future__ import annotations

from typing import Dict, List, Optional

from .models import StateModel, TaskSpec


def initial_tracking(task: TaskSpec) -> StateModel:
    return StateModel(
        task_id=task.task_id,
        step_count=0,
        done=False,
        discovered_keys={ticket.ticket_id: [] for ticket in task.tickets},
        priorities={ticket.ticket_id: None for ticket in task.tickets},
        routes={ticket.ticket_id: None for ticket in task.tickets},
        resolutions={ticket.ticket_id: None for ticket in task.tickets},
        escalations={ticket.ticket_id: None for ticket in task.tickets},
        queue_order=[],
        cumulative_reward=0.0,
        latest_score={},
    )


def update_mapping(
    current: Dict[str, Optional[str]],
    ticket_id: str,
    value: Optional[str],
) -> Dict[str, Optional[str]]:
    current[ticket_id] = value
    return current


def discovered_for_ticket(discovered_keys: Dict[str, List[str]], ticket_id: str) -> List[str]:
    return discovered_keys.setdefault(ticket_id, [])
