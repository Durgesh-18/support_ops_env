from __future__ import annotations

import json
import os

import gradio as gr

from server.app import app as api_app
from support_ops_env.models import Action
from support_ops_env.tasks import list_task_ids


def reset_env(task_id: str) -> str:
    from server.app import _http_env

    observation = _http_env.reset(task_id=task_id)
    return json.dumps(observation.model_dump(), indent=2)


def step_env(task_id: str, action_type: str, target: str, value: str) -> tuple[str, str]:
    from server.app import _http_env

    state = _http_env.state()
    if state.task_id != task_id or (state.step_count == 0 and not state.done):
        _http_env.reset(task_id=task_id)

    action = Action(action_type=action_type, target=target or "T1", value=value or None)
    observation, reward, done, info = _http_env.step(action)
    payload = {
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }
    return json.dumps(observation.model_dump(), indent=2), json.dumps(payload, indent=2)


with gr.Blocks(title="SupportOpsEnv") as demo:
    gr.Markdown("# SupportOpsEnv")
    gr.Markdown("Multi-step support triage benchmark with deterministic graders.")

    task_id = gr.Dropdown(choices=list_task_ids(), value=list_task_ids()[0], label="Task")
    action_type = gr.Dropdown(
        choices=[
            "inspect_ticket",
            "request_context",
            "set_priority",
            "set_route",
            "set_resolution",
            "escalate",
            "rank_queue",
            "finalize",
        ],
        value="inspect_ticket",
        label="Action Type",
    )
    target = gr.Textbox(value="T1", label="Target Ticket")
    value = gr.Textbox(label="Value")
    observation_output = gr.Code(label="Observation", language="json")
    result_output = gr.Code(label="Step Result", language="json")

    reset_button = gr.Button("Reset")
    step_button = gr.Button("Step")

    reset_button.click(reset_env, inputs=[task_id], outputs=[observation_output])
    step_button.click(
        step_env,
        inputs=[task_id, action_type, target, value],
        outputs=[observation_output, result_output],
    )


app = gr.mount_gradio_app(api_app, demo, path="/ui")


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    resolved_port = port or int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    uvicorn.run(app, host=host, port=resolved_port)


if __name__ == "__main__":
    main()
