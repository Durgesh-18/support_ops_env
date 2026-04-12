from dotenv import load_dotenv
load_dotenv()
import json
import os
import re
import textwrap
import time
from typing import List, Optional

from openai import OpenAI

from support_ops_env.env import SupportOpsEnv
from support_ops_env.models import Action, Observation
from support_ops_env.tasks import list_task_ids

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("SUPPORT_OPS_TASK", "easy_account_takeover")
BENCHMARK = os.getenv("SUPPORT_OPS_BENCHMARK", "support_ops_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "24"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))  # reasoning models need budget for <think> blocks
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.8"))

# FIX 1: Retry budget for malformed JSON responses before giving up
JSON_RETRY_LIMIT = int(os.getenv("JSON_RETRY_LIMIT", "3"))

# Minimum number of tasks required by the grader
MIN_TASKS = 3

# Actions that must be completed for every ticket before finalize is allowed.
# finalize without these is the #1 score killer based on the logs.
REQUIRED_PER_TICKET = {"set_priority", "set_route", "set_resolution"}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are operating a customer support triage environment.
    Return exactly one JSON object with keys: action_type, target, value. No extra text, no markdown, no code fences.

    Allowed action_type values:
    - inspect_ticket
    - request_context
    - set_priority
    - set_route
    - set_resolution
    - escalate
    - rank_queue
    - finalize

    VALID VALUES — you MUST use these exact strings:

    priority values: urgent, high, normal, low
    route values: account_security, monetization_compliance, billing_refunds, policy_appeals
    resolution values: temporary_lock_and_manual_recovery, request_tax_renewal, approve_refund, expedited_human_review
    escalation teams: security_specialist (only when account compromise is confirmed; omit otherwise)

    ACTION FORMAT EXAMPLES — copy these exactly:
    {"action_type": "inspect_ticket",   "target": "T1", "value": ""}
    {"action_type": "request_context",  "target": "T1", "value": "tax_status"}
    {"action_type": "set_priority",     "target": "T1", "value": "urgent"}
    {"action_type": "set_route",        "target": "T1", "value": "account_security"}
    {"action_type": "set_resolution",   "target": "T1", "value": "temporary_lock_and_manual_recovery"}
    {"action_type": "escalate",         "target": "T1", "value": "security_specialist"}
    {"action_type": "rank_queue",       "target": "queue", "value": "T2,T1,T3"}
    {"action_type": "finalize",         "target": "T1", "value": ""}

    CRITICAL: For request_context, target = ticket ID (e.g. "T1"), value = context key name.
    NEVER put the context key name in target. target is ALWAYS a ticket ID.

    MANDATORY WORKFLOW — follow in this exact order for each ticket:
    1. inspect_ticket (target=ticket_id, value="")  ← ONCE per ticket, BEFORE any other action on it.
    2. request_context ONLY for keys in required_context_keys (these affect your score).
       Use target=ticket_id, value=key_name. One key per step. Request each key at most once.
       Do NOT request optional available_context_keys — they waste steps.
    3. set_priority  ← MANDATORY before finalize. Use valid priority values.
    4. set_route     ← MANDATORY before finalize. Use valid route values.
    5. set_resolution ← MANDATORY before finalize. Use valid resolution values.
    6. escalate only when account takeover / security compromise is confirmed.
    7. For queue tasks: rank_queue once, after ALL tickets are processed.
    8. finalize (target=ticket_id, value="") — ONLY after set_priority, set_route,
       and set_resolution have ALL been called for this ticket.

    *** YOU MUST call set_priority, set_route, and set_resolution on every ticket. ***
    *** Calling finalize before those three actions will score near 0. ***

    PRIORITY HINTS:
    - Account takeover / fraud / SLA <= 2h → urgent
    - Tax/compliance holds, payment issues / SLA <= 12h → high
    - Routine appeals, refunds / SLA >= 24h → normal

    STRICT RULES:
    - NEVER repeat an action you have already taken (check your history).
    - inspect_ticket AT MOST ONCE per ticket, and ALWAYS before request_context on that ticket.
    - target is ALWAYS a ticket ID like "T1". NEVER put a context key in target.
    - Each request_context must use a different value (key name).
    - value must ALWAYS be a string — use "" (empty string), never null.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(
    observation: Observation,
    step: int,
    rewards: List[float],
    action_history: List[str],
    completed_per_ticket: dict,
) -> str:
    reward_history = ",".join(f"{reward:.2f}" for reward in rewards[-5:]) if rewards else "none"
    history_str = "\n".join(f"  {a}" for a in action_history) if action_history else "  none"

    # FIX 2: Summarise what mandatory actions are still missing per ticket so the
    # model can see at a glance what it still needs to do before finalize.
    pending_lines = []
    for tid, done_actions in sorted(completed_per_ticket.items()):
        missing = REQUIRED_PER_TICKET - done_actions
        if missing:
            pending_lines.append(f"  {tid}: still needs {', '.join(sorted(missing))}")
    pending_str = "\n".join(pending_lines) if pending_lines else "  all mandatory actions complete"

    return textwrap.dedent(
        f"""
        Step: {step}
        Task: {observation.task_id}
        Difficulty: {observation.difficulty}
        Reward history: {reward_history}

        Mandatory actions still PENDING (you MUST complete these before finalize):
{pending_str}

        Actions you have ALREADY taken this episode (do NOT repeat these):
{history_str}

        Observation JSON:
        {json.dumps(observation.model_dump(), indent=2, sort_keys=True)}
        Return one JSON action that you have NOT already taken.
        Remember: value must always be a string, never null.
        """
    ).strip()


def extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from model output.
    Handles:
    - <think>...</think> reasoning blocks (emitted by DeepSeek-R1, Gemini thinking, etc.)
    - Markdown code fences (```json ... ```)
    - Stray surrounding text
    """
    # Strip <think>...</think> blocks first — they often contain stray { } chars
    # that fool the JSON extractor into grabbing the wrong object.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip ```json ... ``` fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the LAST complete {...} block — the real action is always after any
    # preamble text, so the last match is more reliable than the first.
    matches = list(re.finditer(r"\{[^{}]+\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid JSON object found in: {text!r}")


def get_model_action(
    client: OpenAI,
    observation: Observation,
    step: int,
    rewards: List[float],
    action_history: List[str],
    completed_per_ticket: dict,
) -> tuple[Action, Optional[str]]:
    user_prompt = build_user_prompt(observation, step, rewards, action_history, completed_per_ticket)
    last_exc: Optional[str] = None
    content = ""

    for attempt in range(1, JSON_RETRY_LIMIT + 1):
        # Slightly raise temperature on retries so we don't get the same bad output
        temp = TEMPERATURE if attempt == 1 else min(TEMPERATURE + 0.15 * attempt, 1.0)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temp,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            content = (completion.choices[0].message.content or "").strip()
            payload = extract_json(content)

            # FIX 4: Normalise null → "" so the Action model never sees None for value
            if payload.get("value") is None:
                payload["value"] = ""

            action = Action.model_validate(payload)
            return action, None
        except Exception as exc:
            last_exc = str(exc).replace("\n", " ")
            print(f"[WARN] attempt={attempt} parse_error={last_exc!r} content={content!r}", flush=True)

            # FIX 5a: Respect rate-limit retry-after delays instead of hammering the API.
            # The 429 body includes a retryDelay field (e.g. "16s"). Parse and sleep for it
            # so subsequent attempts actually succeed rather than burning the retry budget.
            if "429" in last_exc or "RESOURCE_EXHAUSTED" in last_exc:
                delay_match = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s", last_exc)
                delay = float(delay_match.group(1)) if delay_match else 20.0
                print(f"[WARN] rate-limited; sleeping {delay:.1f}s before retry", flush=True)
                time.sleep(delay)

    # FIX 5b: Exhausted retries — do NOT blindly finalize.
    # Skip to a no-op inspect on the first visible ticket to keep the episode alive.
    print("[WARN] JSON retry limit exhausted; emitting safe no-op", flush=True)
    # observation.tickets may be a list of objects or a dict — handle both.
    obs_dump = observation.model_dump()
    raw_tickets = obs_dump.get("tickets", [])
    if isinstance(raw_tickets, dict):
        ticket_ids = list(raw_tickets.keys())
    else:
        # list of dicts — each item should have an "id" or similar field
        ticket_ids = [
            t.get("ticket_id") or t.get("id") or f"T{i+1}"
            for i, t in enumerate(raw_tickets)
        ]
    ticket_ids = ticket_ids or ["T1"]

    inspected = {
        json.loads(a)["target"]
        for a in action_history
        if json.loads(a).get("action_type") == "inspect_ticket"
    }
    target = next((t for t in ticket_ids if t not in inspected), ticket_ids[0])
    fallback = Action(action_type="inspect_ticket", target=target, value="")
    return fallback, last_exc


def clamp_score(score: float) -> float:
    """Clamp score to strictly open interval (0, 1).
    Uses 0.001/0.999 so the value survives :.3f log formatting —
    the submission parser reads that string, so 1e-6 would round
    to '0.000' and be rejected as exactly 0.0."""
    return min(max(float(score), 0.001), 0.999)


def select_tasks(requested: str) -> List[str]:
    """
    Return at least MIN_TASKS task IDs.
    Always includes the requested task; pads with other available tasks if needed.
    """
    available = list_task_ids()
    if not available:
        raise RuntimeError("No tasks available in the environment.")

    primary = requested if requested in available else available[0]
    others = [t for t in available if t != primary]
    task_list = [primary] + others
    return task_list[:max(MIN_TASKS, 1)]


def run_task(client: OpenAI, task_name: str) -> dict:
    """Run a single task and return a result dict."""
    env = SupportOpsEnv(task_id=task_name)
    rewards: List[float] = []
    action_history: List[str] = []
    # FIX 6: Track which mandatory actions have been completed per ticket
    # so we can warn the model and block premature finalize.
    completed_per_ticket: dict[str, set] = {}
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=task_name)

        for step in range(1, MAX_STEPS + 1):
            action, action_error = get_model_action(
                client, observation, step, rewards, action_history, completed_per_ticket
            )

            # FIX 7: Guard against premature finalize — if mandatory steps are still
            # missing for any ticket, redirect to the first pending mandatory action
            # instead of letting the model throw away the score.
            if action.action_type == "finalize":
                target = action.target or "T1"
                missing = REQUIRED_PER_TICKET - completed_per_ticket.get(target, set())
                if missing:
                    next_action_type = sorted(missing)[0]  # deterministic ordering
                    print(
                        f"[GUARD] Premature finalize on {target}; redirecting to {next_action_type}",
                        flush=True,
                    )
                    # Pick the first valid value for the missing action type
                    FALLBACK_VALUES = {
                        "set_priority": "normal",
                        "set_route": "policy_appeals",
                        "set_resolution": "expedited_human_review",
                    }
                    action = Action(
                        action_type=next_action_type,
                        target=target,
                        value=FALLBACK_VALUES[next_action_type],
                    )

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            action_history.append(action_str)

            # Update completion tracker
            if action.action_type in REQUIRED_PER_TICKET:
                t = action.target or "T1"
                completed_per_ticket.setdefault(t, set()).add(action.action_type)

            observation, reward, done, info = env.step(action)
            reward_value = reward.value
            rewards.append(reward_value)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward_value,
                done=done,
                error=action_error,
            )

            score = float(info.get("task_score", 0.0))
            if done:
                break

        score = clamp_score(score)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "success": success, "steps": steps_taken, "score": score}


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = list(reversed(select_tasks(TASK_NAME)))

    all_results = []
    for task_name in tasks:
        result = run_task(client, task_name)
        all_results.append(result)

    total = len(all_results)
    passed = sum(1 for r in all_results if r["success"])
    avg_score = sum(r["score"] for r in all_results) / total if total else 0.0
    print(
        f"[SUMMARY] tasks={total} passed={passed} avg_score={avg_score:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
