from __future__ import annotations

import unittest

from server.app import (
    StepRequest,
    app,
    health,
    metadata,
    reset,
    schema,
    state,
    step,
)


class ServerApiTest(unittest.TestCase):
    def test_required_routes_are_registered(self) -> None:
        route_paths = {route.path for route in app.routes}
        self.assertIn("/health", route_paths)
        self.assertIn("/metadata", route_paths)
        self.assertIn("/schema", route_paths)
        self.assertIn("/reset", route_paths)
        self.assertIn("/step", route_paths)
        self.assertIn("/state", route_paths)
        self.assertIn("/ws", route_paths)

    def test_handlers_return_openenv_shaped_payloads(self) -> None:
        self.assertEqual(health().status, "healthy")
        self.assertEqual(metadata().name, "support-ops-env")
        self.assertIn("action_type", schema().action["properties"])

        reset_response = reset()
        self.assertEqual(reset_response.observation["task_id"], "easy_account_takeover")
        self.assertFalse(reset_response.done)

        state_response = state()
        self.assertEqual(state_response["task_id"], "easy_account_takeover")
        self.assertIn("episode_id", state_response)

        step_response = step(StepRequest(action={"action_type": "inspect_ticket", "target": "T1"}))
        self.assertIn("observation", step_response.model_dump())
        self.assertIsInstance(step_response.reward, float)


if __name__ == "__main__":
    unittest.main()
