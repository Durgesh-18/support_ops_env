from __future__ import annotations

from typing import Dict, List

from ..models import StateModel, TaskGrade, TaskSpec, TicketSpec

# Scores must be strictly within (0, 1) — the submission grader rejects
# exact 0.0 and 1.0.  Because all non-context components are binary,
# a perfect or zero run would otherwise produce exactly 0.0 or 1.0.
_SCORE_MIN = 1e-6
_SCORE_MAX = 1.0 - 1e-6


def _clamp(score: float) -> float:
    return min(max(score, _SCORE_MIN), _SCORE_MAX)


def _ticket_component(
    ticket: TicketSpec,
    state: StateModel,
    weights: Dict[str, float],
) -> Dict[str, float]:
    discovered = set(state.discovered_keys.get(ticket.ticket_id, []))
    required = set(ticket.required_context)
    context_score = 1.0 if not required else len(discovered & required) / len(required)
    escalation_value = state.escalations.get(ticket.ticket_id)
    gold_escalation = ticket.gold_escalation_team
    escalation_score = 1.0 if escalation_value == gold_escalation else 0.0
    if gold_escalation is None and escalation_value is None:
        escalation_score = 1.0

    raw = {
        "context": context_score,
        "priority": 1.0 if state.priorities.get(ticket.ticket_id) == ticket.gold_priority else 0.0,
        "route": 1.0 if state.routes.get(ticket.ticket_id) == ticket.gold_route else 0.0,
        "resolution": 1.0 if state.resolutions.get(ticket.ticket_id) == ticket.gold_resolution else 0.0,
        "escalation": escalation_score,
    }
    return {name: raw[name] * weights.get(name, 0.0) for name in raw}


def grade_single_ticket(
    task: TaskSpec,
    state: StateModel,
    weights: Dict[str, float],
) -> TaskGrade:
    ticket = task.tickets[0]
    weighted = _ticket_component(ticket, state, weights)
    score = _clamp(round(sum(weighted.values()), 4))
    notes = _notes_for_ticket(ticket, state)
    return TaskGrade(
        task_id=task.task_id,
        score=score,
        passed=score >= 0.8,
        component_scores=weighted,
        notes=notes,
    )


def grade_queue_task(
    task: TaskSpec,
    state: StateModel,
    weights: Dict[str, float],
) -> TaskGrade:
    ticket_scores: List[float] = []
    component_sums = {
        "context": 0.0,
        "priority": 0.0,
        "route": 0.0,
        "resolution": 0.0,
        "escalation": 0.0,
    }
    notes: List[str] = []
    for ticket in task.tickets:
        weighted = _ticket_component(ticket, state, weights)
        for name, value in weighted.items():
            component_sums[name] += value
        ticket_scores.append(sum(weighted.values()))
        notes.extend(_notes_for_ticket(ticket, state))

    divisor = max(len(task.tickets), 1)
    averaged = {name: round(value / divisor, 4) for name, value in component_sums.items()}

    ranking_score = 0.0
    if task.gold_queue_order:
        matches = sum(
            1 for observed, expected in zip(state.queue_order, task.gold_queue_order) if observed == expected
        )
        ranking_score = round((matches / len(task.gold_queue_order)) * weights.get("ranking", 0.0), 4)

    averaged["ranking"] = ranking_score
    score = _clamp(round(sum(averaged.values()), 4))
    return TaskGrade(
        task_id=task.task_id,
        score=score,
        passed=score >= 0.8,
        component_scores=averaged,
        notes=notes,
    )


def _notes_for_ticket(ticket: TicketSpec, state: StateModel) -> List[str]:
    notes: List[str] = []
    if state.priorities.get(ticket.ticket_id) != ticket.gold_priority:
        notes.append(f"{ticket.ticket_id}: incorrect priority")
    if state.routes.get(ticket.ticket_id) != ticket.gold_route:
        notes.append(f"{ticket.ticket_id}: incorrect route")
    if state.resolutions.get(ticket.ticket_id) != ticket.gold_resolution:
        notes.append(f"{ticket.ticket_id}: incorrect resolution")
    if state.escalations.get(ticket.ticket_id) != ticket.gold_escalation_team:
        if not (ticket.gold_escalation_team is None and state.escalations.get(ticket.ticket_id) is None):
            notes.append(f"{ticket.ticket_id}: incorrect escalation")
    missing = set(ticket.required_context) - set(state.discovered_keys.get(ticket.ticket_id, []))
    if missing:
        notes.append(f"{ticket.ticket_id}: missing required context {sorted(missing)}")
    return notes
