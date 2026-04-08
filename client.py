"""Root-level client wrapper for OpenEnv packaging."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import Action, Observation


class SupportOpsEnvClient(EnvClient[Action, Observation, State]):
    def _step_payload(self, action: Action) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        observation = Observation.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


__all__ = ["SupportOpsEnvClient"]
