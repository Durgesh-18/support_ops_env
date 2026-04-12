from dotenv import load_dotenv
load_dotenv()
import json
import os
import textwrap
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
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.8"))

# Minimum number of tasks required by the grader
MIN_TASKS = 3

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are operating a customer support triage environment.
    Return exactly one JSON object with keys: action_type, target, value.

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
    {"action_type": "rank_queue",       "target": "T1", "value": "T2,T1,T3"}
    {"action_type": "finalize",         "target": "T1", "value": ""}

    CRITICAL: For request_context, target = ticket ID (e.g. "T1"), value = context key name.
    NEVER put the context key name in target. target is ALWAYS a ticket ID.

    WORKFLOW PER TICKET:
    1. inspect_ticket once (target=ticket_id, value="").
    2. request_context ONLY for keys in required_context_keys first (these affect your score).
       Use target=ticket_id, value=key_name. Request each key at most once.
       Do NOT request optional keys from available_context_keys — they give tiny reward
       but waste steps you need for set_resolution, escalate, rank_queue, and finalize.
    3. set_priority, set_route, set_resolution using the VALID VALUES listed above.
       Use the context you discovered to choose correctly.
    4. escalate only when account takeover / security compromise is confirmed.
    5. For queue tasks: rank_queue after processing all tickets (most urgent first).
    6. finalize (target=ticket_id, value="") when all tickets are done.

    PRIORITY HINTS:
    - Account takeover / fraud / SLA <= 2h → urgent
    - Tax/compliance holds, payment issues / SLA <= 12h → high
    - Routine appeals, refunds / SLA >= 24h → normal

    STRICT RULES:
    - NEVER repeat an action you have already taken (check your history).
    - inspect_ticket AT MOST ONCE per ticket.
    - target is ALWAYS a ticket ID like "T1". NEVER put a context key in target.
    - Each request_context must use a different value (key name).
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


def build_user_prompt(observation: Observation, step: int, rewards: List[float], action_history: List[str]) -> str:
    reward_history = ",".join(f"{reward:.2f}" for reward in rewards[-5:]) if rewards else "none"
    history_str = "\n".join(f"  {a}" for a in action_history) if action_history else "  none"
    return textwrap.dedent(
        f"""
        Step: {step}
        Task: {observation.task_id}
        Difficulty: {observation.difficulty}
        Reward history: {reward_history}

        Actions you have ALREADY taken this episode (do NOT repeat these):
{history_str}

        Observation JSON:
        {json.dumps(observation.model_dump(), indent=2, sort_keys=True)}
        Return one JSON action that you have NOT already taken.
        """
    ).strip()


def get_model_action(client: OpenAI, observation: Observation, step: int, rewards: List[float], action_history: List[str]) -> tuple[Action, Optional[str]]:
    user_prompt = build_user_prompt(observation, step, rewards, action_history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        payload = json.loads(content)
        action = Action.model_validate(payload)
        return action, None
    except Exception as exc:
        fallback = Action(action_type="finalize")
        return fallback, str(exc).replace("\n", " ")


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

    # Start with the requested task (validated), then fill up to MIN_TASKS
    primary = requested if requested in available else available[0]
    others = [t for t in available if t != primary]
    task_list = [primary] + others
    return task_list[:max(MIN_TASKS, 1)]


def run_task(client: OpenAI, task_name: str) -> dict:
    """Run a single task and return a result dict."""
    env = SupportOpsEnv(task_id=task_name)
    rewards: List[float] = []
    action_history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=task_name)

        for step in range(1, MAX_STEPS + 1):
            action, action_error = get_model_action(client, observation, step, rewards, action_history)
            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            action_history.append(action_str)

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

        # Fix 1: clamp to strictly open (0, 1) — grader rejects 0.0 and 1.0
        score = clamp_score(score)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "success": success, "steps": steps_taken, "score": score}


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Fix 2: run at least MIN_TASKS tasks so the grader has enough scored entries
    # Run in reverse difficulty order (hard first) so expensive tasks get credits
    # while the budget is fresh, rather than always dying on the last task.
    tasks = list(reversed(select_tasks(TASK_NAME)))

    all_results = []
    for task_name in tasks:
        result = run_task(client, task_name)
        all_results.append(result)

    # Summary across all tasks
    total = len(all_results)
    passed = sum(1 for r in all_results if r["success"])
    avg_score = sum(r["score"] for r in all_results) / total if total else 0.0
    print(
        f"[SUMMARY] tasks={total} passed={passed} avg_score={avg_score:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
