from __future__ import annotations

import unittest

from scripts.run_rule_baseline import choose_next_action
from support_ops_env.env import SupportOpsEnv
from support_ops_env.tasks import list_task_ids


class RuleBaselineTest(unittest.TestCase):
    def test_rule_baseline_solves_all_tasks(self) -> None:
        for task_id in list_task_ids():
            env = SupportOpsEnv(task_id=task_id)
            observation = env.reset()
            done = False
            last_info = {}
            while not done:
                action = choose_next_action(observation)
                observation, _, done, info = env.step(action)
                last_info = info
            self.assertAlmostEqual(last_info["task_score"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
