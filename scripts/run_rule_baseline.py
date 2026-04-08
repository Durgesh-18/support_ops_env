from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support_ops_env.env import SupportOpsEnv
from support_ops_env.models import Action, BaselineResult, Observation, TicketObservation
from support_ops_env.tasks import list_task_ids


CONTEXT_PRIORITY = [
    "account_security",
    "billing_activity",
    "tax_status",
    "payout_hold",
    "appeal_state",
    "campaign_deadline",
    "payment_status",
]


def choose_next_action(observation: Observation) -> Action:
    if observation.queue_mode and not observation.current_queue_order:
        ranking = rank_tickets(observation.tickets)
        return Action(action_type="rank_queue", value=",".join(ranking))

    for ticket in observation.tickets:
        next_context = missing_high_value_context(ticket)
        if next_context:
            return Action(action_type="request_context", target=ticket.ticket_id, value=next_context)

    for ticket in observation.tickets:
        priority = infer_priority(ticket)
        if ticket.selected_priority != priority:
            return Action(action_type="set_priority", target=ticket.ticket_id, value=priority)

    for ticket in observation.tickets:
        route = infer_route(ticket)
        if ticket.selected_route != route:
            return Action(action_type="set_route", target=ticket.ticket_id, value=route)

    for ticket in observation.tickets:
        resolution = infer_resolution(ticket)
        if ticket.selected_resolution != resolution:
            return Action(action_type="set_resolution", target=ticket.ticket_id, value=resolution)

    for ticket in observation.tickets:
        escalation = infer_escalation(ticket)
        if ticket.escalation_team != escalation:
            return Action(action_type="escalate", target=ticket.ticket_id, value=escalation)

    return Action(action_type="finalize")


def missing_high_value_context(ticket: TicketObservation) -> str | None:
    discovered = set(ticket.discovered_context)
    haystack = flattened_text(ticket)

    candidates: List[str] = infer_required_context(ticket)

    for key in CONTEXT_PRIORITY:
        if key in candidates and key not in discovered:
            return key
    return None


def infer_required_context(ticket: TicketObservation) -> List[str]:
    text = flattened_text(ticket)
    if "payout" in text or "w-9" in text or "bank details" in text or "funds released" in text:
        return ["tax_status", "payout_hold"]
    if "appeal" in text or "auto-removed" in text or "monetization is paused" in text:
        return ["appeal_state", "campaign_deadline"]
    if "duplicate charge" in text or "refund" in text:
        return ["payment_status"]
    if (
        "login" in text
        or "ad spend" in text
        or "unfamiliar campaigns" in text
        or "taken over" in text
        or "recovery email was changed" in text
    ):
        return ["account_security", "billing_activity"]
    return []


def infer_priority(ticket: TicketObservation) -> str:
    text = flattened_text(ticket)
    if (
        "critical" in text
        or "$1,900" in text
        or "unauthorized ad spend" in text
        or "impossible travel" in text
        or "recovery email was changed" in text
    ):
        return "urgent"
    if "campaign begins in 18 hours" in text or "monetization is paused" in text:
        return "high"
    if "w-9 expired" in text or "monthly payout" in text:
        return "high"
    return "normal"


def infer_route(ticket: TicketObservation) -> str:
    text = flattened_text(ticket)
    if (
        "account takeover" in text
        or "new devices" in text
        or "recovery email was changed" in text
        or "unfamiliar campaigns" in text
        or "unauthorized ad spend" in text
        or "losing access" in text
    ):
        return "account_security"
    if "w-9 expired" in text or "compliance hold" in text:
        return "monetization_compliance"
    if "auto-removed" in text or "human yet" in text:
        return "policy_appeals"
    if "duplicate charge" in text or "automatically refundable" in text:
        return "billing_refunds"
    return "general_support"


def infer_resolution(ticket: TicketObservation) -> str:
    text = flattened_text(ticket)
    if (
        "account takeover" in text
        or "new devices" in text
        or "impossible travel" in text
        or "unfamiliar campaigns" in text
        or "losing access" in text
    ):
        return "temporary_lock_and_manual_recovery"
    if "w-9 expired" in text or "compliance hold" in text:
        return "request_tax_renewal"
    if "auto-removed" in text or "sponsored campaign begins" in text:
        return "expedited_human_review"
    if "duplicate charge" in text or "automatically refundable" in text:
        return "approve_refund"
    return "request_more_info"


def infer_escalation(ticket: TicketObservation) -> str | None:
    text = flattened_text(ticket)
    if (
        "account takeover" in text
        or "critical" in text
        or "impossible travel" in text
        or "unfamiliar campaigns" in text
        or "losing access" in text
    ):
        return "security_specialist"
    return None


def rank_tickets(tickets: List[TicketObservation]) -> List[str]:
    scored = []
    for ticket in tickets:
        text = flattened_text(ticket)
        score = 0
        if "critical" in text or "account takeover" in text or "$1,900" in text or "unfamiliar campaigns" in text:
            score += 100
        if "campaign begins in 18 hours" in text or "sponsored campaign" in text:
            score += 60
        if "duplicate charge" in text:
            score += 20
        if ticket.visible_context.get("sla_hours_remaining") == "1":
            score += 30
        if ticket.visible_context.get("sla_hours_remaining") == "4":
            score += 10
        scored.append((score, ticket.ticket_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [ticket_id for _, ticket_id in scored]


def flattened_text(ticket: TicketObservation) -> str:
    parts = [
        ticket.summary,
        json.dumps(ticket.visible_context, sort_keys=True),
        json.dumps(ticket.discovered_context, sort_keys=True),
    ]
    return " ".join(parts).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic rule-based baseline over all tasks.")
    parser.add_argument("--output", default="rule_baseline_results.json", help="Path to write JSON results")
    args = parser.parse_args()

    results: List[BaselineResult] = []
    for task_id in list_task_ids():
        env = SupportOpsEnv(task_id=task_id)
        observation = env.reset()
        done = False
        transcript: List[Dict[str, object]] = []
        last_info: Dict[str, object] = {}

        while not done:
            action = choose_next_action(observation)
            observation, reward, done, info = env.step(action)
            transcript.append(
                {
                    "action": action.model_dump(),
                    "reward": reward.model_dump(),
                    "task_score": info["task_score"],
                    "done": done,
                }
            )
            last_info = info

        results.append(
            BaselineResult(
                task_id=task_id,
                difficulty=observation.difficulty,
                score=float(last_info.get("task_score", 0.0)),
                steps=int(last_info.get("step_count", 0)),
                transcript=transcript,
            )
        )

    payload = {
        "baseline": "rule_based",
        "average_score": round(sum(item.score for item in results) / len(results), 4),
        "results": [item.model_dump() for item in results],
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
