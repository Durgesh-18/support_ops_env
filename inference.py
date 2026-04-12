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
MAX_STEPS = int(os.getenv("MAX_STEPS", "16"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.8"))


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
    Choose only valid ticket ids from the observation.
    Use concise string values.
    Finalize only after enough evidence is gathered.
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


def build_user_prompt(observation: Observation, step: int, rewards: List[float]) -> str:
    reward_history = ",".join(f"{reward:.2f}" for reward in rewards[-5:]) if rewards else "none"
    return textwrap.dedent(
        f"""
        Step: {step}
        Task: {observation.task_id}
        Difficulty: {observation.difficulty}
        Reward history: {reward_history}
        Observation JSON:
        {json.dumps(observation.model_dump(), indent=2, sort_keys=True)}
        Return one JSON action.
        """
    ).strip()


def get_model_action(client: OpenAI, observation: Observation, step: int, rewards: List[float]) -> tuple[Action, Optional[str]]:
    user_prompt = build_user_prompt(observation, step, rewards)
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


def ensure_known_task(task_name: str) -> str:
    if task_name in list_task_ids():
        return task_name
    return list_task_ids()[0]


def main() -> None:
    task_name = ensure_known_task(TASK_NAME)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SupportOpsEnv(task_id=task_name)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=task_name)

        for step in range(1, MAX_STEPS + 1):
            action, action_error = get_model_action(client, observation, step, rewards)
            action_str = json.dumps(action.model_dump(), separators=(",", ":"))

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

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
