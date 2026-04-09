from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from support_ops_env.env import SupportOpsEnv
from support_ops_env.models import Action, Observation, StateModel


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: Optional[int] = Field(default=None, ge=0)
    episode_id: Optional[str] = None
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: Dict[str, Any]
    timeout_s: Optional[float] = Field(default=None, gt=0)
    request_id: Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class HealthResponse(BaseModel):
    status: str = "healthy"


class SchemaResponse(BaseModel):
    action: Dict[str, Any]
    observation: Dict[str, Any]
    state: Dict[str, Any]


class EnvironmentMetadata(BaseModel):
    name: str
    description: str
    readme_content: Optional[str] = None
    version: str
    documentation_url: Optional[str] = None


README_PATH = Path(__file__).resolve().parent.parent / "README.md"

app = FastAPI(
    title="SupportOpsEnv Server",
    description="OpenEnv-compatible server for the SupportOpsEnv benchmark.",
    version="0.1.0",
)

_http_env = SupportOpsEnv()
_http_episode_id = str(uuid4())


def _serialize_observation(
    observation: Observation,
    reward: Optional[float] = None,
    done: bool = False,
) -> Dict[str, Any]:
    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
    }


def _state_payload(env: SupportOpsEnv, episode_id: str) -> Dict[str, Any]:
    state = env.state().model_dump()
    state["episode_id"] = episode_id
    return state


def _metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="support-ops-env",
        description="Multi-step customer support triage and escalation benchmark for OpenEnv-style agents.",
        readme_content=README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else None,
        version="0.1.0",
        documentation_url="https://huggingface.co/spaces",
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "support-ops-env",
        "status": "ok",
        "message": "SupportOpsEnv OpenEnv server is available.",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/metadata", response_model=EnvironmentMetadata)
def metadata() -> EnvironmentMetadata:
    return _metadata()


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    return SchemaResponse(
        action=Action.model_json_schema(),
        observation=Observation.model_json_schema(),
        state=StateModel.model_json_schema(),
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    return _state_payload(_http_env, _http_episode_id)


@app.post("/reset", response_model=StepResponse)
def reset(request: ResetRequest = ResetRequest()) -> StepResponse:
    global _http_episode_id

    _http_episode_id = request.episode_id or str(uuid4())
    observation = _http_env.reset(task_id=request.task_id)
    return StepResponse(**_serialize_observation(observation))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    try:
        action = Action.model_validate(request.action)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    observation, reward, done, _info = _http_env.step(action)
    return StepResponse(
        **_serialize_observation(observation, reward=reward.value, done=done)
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    env = SupportOpsEnv()
    episode_id = str(uuid4())

    try:
        while True:
            raw_message = await websocket.receive_text()

            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {"message": f"Invalid JSON: {exc}", "code": "invalid_json"},
                    }
                )
                continue

            msg_type = payload.get("type")
            data = payload.get("data", {})

            try:
                if msg_type == "reset":
                    reset_request = ResetRequest.model_validate(data)
                    episode_id = reset_request.episode_id or str(uuid4())
                    observation = env.reset(task_id=reset_request.task_id)
                    await websocket.send_json(
                        {"type": "observation", "data": _serialize_observation(observation)}
                    )
                elif msg_type == "step":
                    action = Action.model_validate(data)
                    observation, reward, done, _info = env.step(action)
                    await websocket.send_json(
                        {
                            "type": "observation",
                            "data": _serialize_observation(
                                observation,
                                reward=reward.value,
                                done=done,
                            ),
                        }
                    )
                elif msg_type == "state":
                    await websocket.send_json(
                        {"type": "state", "data": _state_payload(env, episode_id)}
                    )
                elif msg_type == "close":
                    break
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "data": {
                                "message": f"Unknown message type: {msg_type}",
                                "code": "unknown_type",
                            },
                        }
                    )
            except ValidationError as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {
                            "message": "Validation error",
                            "code": "validation_error",
                            "errors": exc.errors(),
                        },
                    }
                )
            except Exception as exc:  # pragma: no cover
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {"message": str(exc), "code": "execution_error"},
                    }
                )
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def uv_main() -> FastAPI:
    return app


if __name__ == "__main__":
    main()
