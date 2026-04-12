from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .graders import grade_task
from .models import Action, Observation, RewardModel, StateModel, StepInfo, TaskSpec, TicketObservation
from .reward import STEP_PENALTY, build_reward
from .state import discovered_for_ticket, initial_tracking, update_mapping
from .tasks import get_all_tasks, get_task


class SupportOpsEnv:
    """OpenEnv-shaped benchmark for support operations workflows."""

    def __init__(self, task_id: Optional[str] = None):
        self._tasks = {task.task_id: task for task in get_all_tasks()}
        self._task_order = sorted(self._tasks)
        self._task_id = task_id or self._task_order[0]
        self._task: TaskSpec = self._tasks[self._task_id]
        self._state: StateModel = initial_tracking(self._task)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is not None:
            self._task = get_task(task_id)
            self._task_id = task_id
        self._state = initial_tracking(self._task)
        return self._build_observation()

    def state(self) -> StateModel:
        return self._state.model_copy(deep=True)

    def step(self, action: Action) -> Tuple[Observation, RewardModel, bool, Dict[str, object]]:
        if self._state.done:
            reward = build_reward({"invalid_after_done": -0.1}, "Episode already finished.")
            info = StepInfo(
                task_id=self._task.task_id,
                step_count=self._state.step_count,
                task_score=self._state.latest_score.get("task_score", 0.0),
                done_reason="already_done",
                event="invalid_after_done",
                event_score=reward.components,
            )
            return self._build_observation(), reward, True, info.model_dump()

        self._state.step_count += 1
        event_scores: Dict[str, float] = {"step_penalty": STEP_PENALTY}
        event_name = action.action_type
        done_reason = None

        if action.action_type == "inspect_ticket":
            event_scores.update(self._handle_inspect(action))
        elif action.action_type == "request_context":
            event_scores.update(self._handle_request_context(action))
        elif action.action_type == "set_priority":
            event_scores.update(self._handle_priority(action))
        elif action.action_type == "set_route":
            event_scores.update(self._handle_route(action))
        elif action.action_type == "set_resolution":
            event_scores.update(self._handle_resolution(action))
        elif action.action_type == "escalate":
            event_scores.update(self._handle_escalation(action))
        elif action.action_type == "rank_queue":
            event_scores.update(self._handle_rank_queue(action))
        elif action.action_type == "finalize":
            self._state.done = True
            done_reason = "agent_finalized"
            grade = grade_task(self._task, self._state)
            self._state.latest_score = {"task_score": grade.score, **grade.component_scores}
            event_scores["terminal_grade"] = grade.score
            reward = build_reward(event_scores, "Final task grade applied.")
            self._state.cumulative_reward = round(self._state.cumulative_reward + reward.value, 4)
            info = StepInfo(
                task_id=self._task.task_id,
                step_count=self._state.step_count,
                task_score=grade.score,
                done_reason=done_reason,
                grade=grade,
                event=event_name,
                event_score=reward.components,
            )
            return self._build_observation(), reward, True, info.model_dump()
        else:
            event_scores["invalid_action"] = -0.1
            event_name = "invalid_action"

        grade = grade_task(self._task, self._state)
        self._state.latest_score = {"task_score": grade.score, **grade.component_scores}

        if self._state.step_count >= self._task.max_steps and not self._state.done:
            self._state.done = True
            done_reason = "max_steps"
            event_scores["timeout_grade"] = grade.score

        reward = build_reward(event_scores, f"Processed {event_name}.")
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward.value, 4)
        info = StepInfo(
            task_id=self._task.task_id,
            step_count=self._state.step_count,
            task_score=grade.score,
            done_reason=done_reason,
            grade=grade if self._state.done else None,
            event=event_name,
            event_score=reward.components,
        )
        return self._build_observation(), reward, self._state.done, info.model_dump()

    def _build_observation(self) -> Observation:
        tickets: List[TicketObservation] = []
        for ticket in self._task.tickets:
            keys = self._state.discovered_keys.get(ticket.ticket_id, [])
            discovered_context = {key: ticket.hidden_context[key] for key in keys}
            available_keys = [k for k in ticket.hidden_context if k not in keys]
            tickets.append(
                TicketObservation(
                    ticket_id=ticket.ticket_id,
                    summary=ticket.summary,
                    visible_context=ticket.visible_context,
                    discovered_context=discovered_context,
                    available_context_keys=available_keys,
                    required_context_keys=[k for k in ticket.required_context if k not in keys],
                    selected_priority=self._state.priorities.get(ticket.ticket_id),
                    selected_route=self._state.routes.get(ticket.ticket_id),
                    selected_resolution=self._state.resolutions.get(ticket.ticket_id),
                    escalation_team=self._state.escalations.get(ticket.ticket_id),
                )
            )

        return Observation(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            title=self._task.title,
            instruction=self._task.instruction,
            queue_mode=self._task.queue_mode,
            tickets=tickets,
            remaining_steps=max(self._task.max_steps - self._state.step_count, 0),
            available_actions=[
                "inspect_ticket",
                "request_context",
                "set_priority",
                "set_route",
                "set_resolution",
                "escalate",
                "rank_queue",
                "finalize",
            ],
            current_queue_order=self._state.queue_order,
            score_hint=self._state.latest_score,
        )

    def _find_ticket(self, ticket_id: str):
        for ticket in self._task.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _handle_inspect(self, action: Action) -> Dict[str, float]:
        ticket = self._find_ticket(action.target)
        if ticket is None:
            return {"invalid_ticket": -0.1}
        key = f"inspected::{ticket.ticket_id}"
        notes = self._state.latest_score.setdefault("inspections", 0.0)
        if notes and key in self._state.latest_score:
            return {"redundant_inspect": -0.03}
        self._state.latest_score[key] = 1.0
        return {"inspect": 0.03}

    def _handle_request_context(self, action: Action) -> Dict[str, float]:
        ticket = self._find_ticket(action.target)
        if ticket is None or not action.value:
            return {"invalid_context_request": -0.1}
        if action.value not in ticket.hidden_context:
            return {"unknown_context_key": -0.08}

        discovered = discovered_for_ticket(self._state.discovered_keys, ticket.ticket_id)
        if action.value in discovered:
            return {"redundant_context_request": -0.05}

        discovered.append(action.value)
        if action.value in ticket.required_context:
            return {"required_context_found": 0.12}
        return {"optional_context_found": 0.04}

    def _handle_priority(self, action: Action) -> Dict[str, float]:
        ticket = self._find_ticket(action.target)
        if ticket is None or not action.value:
            return {"invalid_priority": -0.1}
        current = self._state.priorities.get(ticket.ticket_id)
        update_mapping(self._state.priorities, ticket.ticket_id, action.value)
        if action.value == current:
            return {"redundant_priority": -0.03}
        return {"priority_match": 0.08 if action.value == ticket.gold_priority else -0.04}

    def _handle_route(self, action: Action) -> Dict[str, float]:
        ticket = self._find_ticket(action.target)
        if ticket is None or not action.value:
            return {"invalid_route": -0.1}
        current = self._state.routes.get(ticket.ticket_id)
        update_mapping(self._state.routes, ticket.ticket_id, action.value)
        if action.value == current:
            return {"redundant_route": -0.03}
        return {"route_match": 0.1 if action.value == ticket.gold_route else -0.06}

    def _handle_resolution(self, action: Action) -> Dict[str, float]:
        ticket = self._find_ticket(action.target)
        if ticket is None or not action.value:
            return {"invalid_resolution": -0.1}
        current = self._state.resolutions.get(ticket.ticket_id)
        update_mapping(self._state.resolutions, ticket.ticket_id, action.value)
        if action.value == current:
            return {"redundant_resolution": -0.03}
        return {"resolution_match": 0.12 if action.value == ticket.gold_resolution else -0.08}

    def _handle_escalation(self, action: Action) -> Dict[str, float]:
        ticket = self._find_ticket(action.target)
        if ticket is None:
            return {"invalid_escalation": -0.1}
        team = action.value
        current = self._state.escalations.get(ticket.ticket_id)
        update_mapping(self._state.escalations, ticket.ticket_id, team)
        if team == current:
            return {"redundant_escalation": -0.03}

        if team == ticket.gold_escalation_team:
            return {"correct_escalation": 0.1}
        if ticket.gold_escalation_team is None and team is None:
            return {"correct_no_escalation": 0.03}
        return {"incorrect_escalation": -0.1}

    def _handle_rank_queue(self, action: Action) -> Dict[str, float]:
        if not self._task.queue_mode or not action.value:
            return {"invalid_queue_ranking": -0.1}
        ranked = [item.strip() for item in action.value.split(",") if item.strip()]
        valid_ticket_ids = {ticket.ticket_id for ticket in self._task.tickets}
        if set(ranked) != valid_ticket_ids:
            return {"malformed_queue_ranking": -0.08}
        self._state.queue_order = ranked
        correct_positions = sum(
            1 for observed, expected in zip(ranked, self._task.gold_queue_order) if observed == expected
        )
        return {"queue_progress": round((correct_positions / len(ranked)) * 0.12, 4)}
