from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import BaseModel

from support_ops_env.env import SupportOpsEnv
from support_ops_env.models import Action, Observation, RewardModel


class OpenEnvSpecTest(unittest.TestCase):
    def test_models_are_pydantic(self) -> None:
        self.assertTrue(issubclass(Observation, BaseModel))
        self.assertTrue(issubclass(Action, BaseModel))
        self.assertTrue(issubclass(RewardModel, BaseModel))

    def test_reset_and_step_shapes(self) -> None:
        env = SupportOpsEnv()
        observation = env.reset()
        self.assertIsInstance(observation, Observation)
        next_observation, reward, done, info = env.step(Action(action_type="inspect_ticket"))
        self.assertIsInstance(next_observation, Observation)
        self.assertIsInstance(reward, RewardModel)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_state_model_exists(self) -> None:
        env = SupportOpsEnv()
        self.assertEqual(env.state().task_id, env.reset().task_id)

    def test_openenv_metadata_file_exists(self) -> None:
        path = Path(__file__).resolve().parent.parent / "openenv.yaml"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("name: support-ops-env", text)
        self.assertIn("openenv", text)


if __name__ == "__main__":
    unittest.main()
