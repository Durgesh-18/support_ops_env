from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
    "inspect_ticket",
    "request_context",
    "set_priority",
    "set_route",
    "set_resolution",
    "escalate",
    "rank_queue",
    "finalize",
]


class RewardModel(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class Action(BaseModel):
    action_type: ActionType
    target: str = "T1"
    value: Optional[str] = None


class TicketObservation(BaseModel):
    ticket_id: str
    summary: str
    visible_context: Dict[str, str]
    discovered_context: Dict[str, str] = Field(default_factory=dict)
    selected_priority: Optional[str] = None
    selected_route: Optional[str] = None
    selected_resolution: Optional[str] = None
    escalation_team: Optional[str] = None


class Observation(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    instruction: str
    queue_mode: bool
    tickets: List[TicketObservation]
    remaining_steps: int
    available_actions: List[str]
    current_queue_order: List[str] = Field(default_factory=list)
    score_hint: Dict[str, float] = Field(default_factory=dict)


class StateModel(BaseModel):
    task_id: str
    step_count: int
    done: bool
    discovered_keys: Dict[str, List[str]]
    priorities: Dict[str, Optional[str]]
    routes: Dict[str, Optional[str]]
    resolutions: Dict[str, Optional[str]]
    escalations: Dict[str, Optional[str]]
    queue_order: List[str]
    cumulative_reward: float
    latest_score: Dict[str, float] = Field(default_factory=dict)


class TicketSpec(BaseModel):
    ticket_id: str
    summary: str
    visible_context: Dict[str, str]
    hidden_context: Dict[str, str]
    required_context: List[str]
    gold_priority: str
    gold_route: str
    gold_resolution: str
    gold_escalation_team: Optional[str] = None


class TaskSpec(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    description: str
    instruction: str
    max_steps: int
    queue_mode: bool = False
    tickets: List[TicketSpec]
    gold_queue_order: List[str] = Field(default_factory=list)
    grader_name: str
    reward_weights: Dict[str, float] = Field(default_factory=dict)


class TaskGrade(BaseModel):
    task_id: str
    score: float
    passed: bool
    component_scores: Dict[str, float]
    notes: List[str] = Field(default_factory=list)


class StepInfo(BaseModel):
    task_id: str
    step_count: int
    task_score: float
    done_reason: Optional[str] = None
    grade: Optional[TaskGrade] = None
    event: str = ""
    event_score: Dict[str, float] = Field(default_factory=dict)


class BaselineResult(BaseModel):
    task_id: str
    difficulty: str
    score: float
    steps: int
    transcript: List[Dict[str, Any]]
