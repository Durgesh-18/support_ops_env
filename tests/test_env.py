from __future__ import annotations

import unittest

from support_ops_env.env import SupportOpsEnv
from support_ops_env.models import Action


class SupportOpsEnvTest(unittest.TestCase):
    def test_easy_task_can_score_perfectly(self) -> None:
        env = SupportOpsEnv("easy_account_takeover")
        env.reset()
        env.step(Action(action_type="request_context", target="T1", value="account_security"))
        env.step(Action(action_type="request_context", target="T1", value="billing_activity"))
        env.step(Action(action_type="set_priority", target="T1", value="urgent"))
        env.step(Action(action_type="set_route", target="T1", value="account_security"))
        env.step(
            Action(
                action_type="set_resolution",
                target="T1",
                value="temporary_lock_and_manual_recovery",
            )
        )
        _, _, done, info = env.step(
            Action(action_type="escalate", target="T1", value="security_specialist")
        )
        self.assertFalse(done)
        _, _, done, info = env.step(Action(action_type="finalize"))
        self.assertTrue(done)
        self.assertAlmostEqual(info["task_score"], 1.0, places=4)

    def test_hard_queue_ranking_is_scored(self) -> None:
        env = SupportOpsEnv("hard_queue_triage")
        env.reset()
        _, _, _, info = env.step(Action(action_type="rank_queue", value="T2,T3,T1"))
        self.assertGreater(info["task_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
