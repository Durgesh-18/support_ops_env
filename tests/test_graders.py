from __future__ import annotations

import unittest

from support_ops_env.graders import grade_task
from support_ops_env.state import initial_tracking
from support_ops_env.tasks import get_task


class GraderTest(unittest.TestCase):
    def test_incomplete_state_scores_below_perfect(self) -> None:
        task = get_task("medium_payout_hold")
        state = initial_tracking(task)
        grade = grade_task(task, state)
        self.assertLess(grade.score, 1.0)
        self.assertGreaterEqual(grade.score, 0.0)

    def test_queue_grader_rewards_ranking(self) -> None:
        task = get_task("hard_queue_triage")
        state = initial_tracking(task)
        state.queue_order = ["T2", "T3", "T1"]
        grade = grade_task(task, state)
        self.assertGreater(grade.component_scores["ranking"], 0.0)


if __name__ == "__main__":
    unittest.main()
