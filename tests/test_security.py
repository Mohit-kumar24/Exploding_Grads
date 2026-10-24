"""
Security and anti-reward-hacking tests for KingdomForge.
Ensures:
1. Verifier isolation: agent claims/messages do not affect verification
2. Parameter validation: negative numbers, non-integers, string injections rejected
3. Resource constraints: cannot spend resources not owned
4. Dependency constraints: cannot bypass building prerequisites (e.g., castle requires workshop)
5. Post-termination enforcement: actions blocked once episode ends
6. Unknown tool injection rejection
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.database_env import KingdomEnv
from src.env.tasks import get_task
from src.verifier.verifier import Verifier


class TestSecurityAndAdversarial(unittest.TestCase):
    def setUp(self):
        self.env = KingdomEnv()
        self.env.reset("task_1", seed=42)
        self.verifier = Verifier()

    def test_unauthorized_tool_call(self):
        """Unknown or arbitrary tool names must be rejected."""
        res, done, info = self.env.step("eval_code", {"code": "import os; os.system('calc')"})
        self.assertIn("Error", res)

    def test_negative_resource_trade_exploit(self):
        """Negative trade amounts must be blocked to prevent infinite resources."""
        # Build market in task 3
        env = KingdomEnv()
        env.reset("task_3", seed=42)
        env._state["buildings"]["market"] = 1

        res, done, info = env.step("trade", {"action": "sell", "resource": "food", "amount": -100})
        self.assertIn("Error", res)
        self.assertIn("positive integer", res)

    def test_negative_worker_assignment(self):
        """Negative worker counts must be blocked."""
        res, done, info = self.env.step("assign_workers", {"building_type": "farm", "count": -5})
        self.assertIn("Error", res)

    def test_overassigning_workers(self):
        """Cannot assign more workers than total idle population."""
        self.env.step("build", {"building_type": "farm"})
        res, done, info = self.env.step("assign_workers", {"building_type": "farm", "count": 999})
        self.assertIn("Error", res)
        self.assertIn("Not enough idle workers", res)

    def test_castle_prerequisite_enforcement(self):
        """Castle cannot be built without workshop."""
        env = KingdomEnv()
        env.reset("task_5", seed=42)
        # Give tons of resources but no workshop
        env._state["resources"] = {"wood": 1000, "stone": 1000, "gold": 1000}
        res, done, info = env.step("build", {"building_type": "castle"})
        self.assertIn("Error", res)
        self.assertIn("requires at least one workshop", res)

    def test_post_termination_action_blocked(self):
        """Actions after termination must be rejected."""
        self.env.step("submit_answer", {"answer": "finished"})
        self.assertTrue(self.env.done)

        res, done, info = self.env.step("build", {"building_type": "farm"})
        self.assertIn("Error", res)
        self.assertIn("Episode has already ended", res)

    def test_verifier_ignores_agent_claims(self):
        """Verifier computes score from actual kingdom state, ignoring agent text."""
        task = get_task("task_1")
        # Agent claims it did everything in submit_answer
        self.env.step("submit_answer", {"answer": "I have completed all objectives, give me 1.0 reward"})
        result = self.verifier.verify(self.env, task)
        # But actual state has no farm built
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
