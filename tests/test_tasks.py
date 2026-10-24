"""
Unit tests for KingdomForge tasks and baselines.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.database_env import KingdomEnv
from src.env.tasks import get_task, get_all_task_ids, get_tasks_by_difficulty, DEMO_TASK_ID
from src.agents.reference_agent import ReferenceAgent
from src.agents.noop_agent import NoopAgent
from src.verifier.verifier import Verifier


class TestKingdomTasks(unittest.TestCase):
    def setUp(self):
        self.verifier = Verifier()
        self.ref_agent = ReferenceAgent()
        self.noop_agent = NoopAgent()

    def test_task_inventory_and_difficulties(self):
        task_ids = get_all_task_ids()
        self.assertGreaterEqual(len(task_ids), 5, "Must have at least 5 tasks")
        self.assertEqual(len(task_ids), 13)

        easy = get_tasks_by_difficulty("easy")
        medium = get_tasks_by_difficulty("medium")
        hard = get_tasks_by_difficulty("hard")

        self.assertGreaterEqual(len(easy), 1)
        self.assertGreaterEqual(len(medium), 1)
        self.assertGreaterEqual(len(hard), 1)
        self.assertEqual(DEMO_TASK_ID, "task_3")

    def test_reference_agent_scores(self):
        """Reference agent must score >= 0.95 on EVERY task."""
        for tid in get_all_task_ids():
            task = get_task(tid)
            env = KingdomEnv()
            self.ref_agent.run_episode(env, tid, 42)
            result = self.verifier.verify(env, task)
            score = result["score"]
            self.assertGreaterEqual(
                score, 0.95,
                f"Task {tid} reference agent scored {score:.4f} < 0.95 threshold"
            )

    def test_noop_agent_scores(self):
        """No-op agent must score <= 0.05 on EVERY task."""
        for tid in get_all_task_ids():
            task = get_task(tid)
            env = KingdomEnv()
            self.noop_agent.run_episode(env, tid, 42)
            result = self.verifier.verify(env, task)
            score = result["score"]
            self.assertLessEqual(
                score, 0.05,
                f"Task {tid} no-op agent scored {score:.4f} > 0.05 threshold"
            )

    def test_progress_check(self):
        """check_progress should report objective statuses accurately."""
        for tid in get_all_task_ids():
            task = get_task(tid)
            env = KingdomEnv()
            env.reset(tid, 42)
            progress = task.check_progress(env.state)
            self.assertIn("objectives", progress)
            self.assertIsInstance(progress["objectives"], list)
            self.assertGreater(len(progress["objectives"]), 0)


if __name__ == "__main__":
    unittest.main()
