from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from support_ops_env.env import SupportOpsEnv
from support_ops_env.models import Action, BaselineResult
from support_ops_env.tasks import list_task_ids


SYSTEM_PROMPT = """You are evaluating a support operations environment.
Return exactly one JSON object with keys: action_type, target, value.
Choose from action_type:
- inspect_ticket
- request_context
- set_priority
- set_route
- set_resolution
- escalate
- rank_queue
- finalize
Be concise and deterministic. Only use ticket ids that appear in the observation.
When enough evidence is gathered, finalize."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible baseline over all SupportOpsEnv tasks.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--output", default="baseline_results.json", help="Path to write JSON results")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")

    client = OpenAI(api_key=api_key)
    results: List[BaselineResult] = []

    for task_id in list_task_ids():
        env = SupportOpsEnv(task_id=task_id)
        observation = env.reset()
        done = False
        transcript: List[Dict[str, object]] = []
        last_info: Dict[str, object] = {}

        while not done:
            response = client.responses.create(
                model=args.model,
                temperature=0,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(observation.model_dump(), indent=2, sort_keys=True),
                    },
                ],
            )
            raw = response.output_text.strip()
            payload = json.loads(raw)
            action = Action.model_validate(payload)
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

    output_path = Path(args.output)
    payload = {
        "model": args.model,
        "average_score": round(sum(item.score for item in results) / len(results), 4),
        "results": [item.model_dump() for item in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
