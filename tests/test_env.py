"""
Unit tests for KingdomForge Environment (KingdomEnv).
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.database_env import KingdomEnv, BUILDING_DEFS, TRADE_RATES


class TestKingdomEnv(unittest.TestCase):
    def setUp(self):
        self.env = KingdomEnv()
        self.env.reset(task_id="task_1", seed=42)

    def test_initial_state(self):
        state = self.env.state
        self.assertIn("resources", state)
        self.assertIn("population", state)
        self.assertIn("buildings", state)
        self.assertIn("workers", state)
        self.assertEqual(state["turn"], 1)
        self.assertEqual(self.env.step_count, 0)
        self.assertFalse(self.env.done)

    def test_state_hashing(self):
        h1 = self.env.state_hash()
        self.assertIsInstance(h1, str)
        self.assertEqual(len(h1), 8)

        # Running same seed gives same hash
        env2 = KingdomEnv()
        env2.reset(task_id="task_1", seed=42)
        h2 = env2.state_hash()
        self.assertEqual(h1, h2)

    def test_build_structure(self):
        # Initial: gold 30, wood 20
        # farm costs wood 10, gold 5
        res, done, info = self.env.step("build", {"building_type": "farm"})
        self.assertIn("Built farm successfully", res)
        self.assertEqual(self.env.state["buildings"].get("farm", 0), 1)
        self.assertEqual(self.env.state["resources"]["wood"], 10)
        self.assertEqual(self.env.state["resources"]["gold"], 25)

    def test_build_insufficient_resources(self):
        # Castle costs wood 50, stone 50, gold 100
        res, done, info = self.env.step("build", {"building_type": "castle"})
        self.assertIn("Error", res)

    def test_assign_workers(self):
        # Build farm first
        self.env.step("build", {"building_type": "farm"})
        # Assign 2 workers
        res, done, info = self.env.step("assign_workers", {"building_type": "farm", "count": 2})
        self.assertIn("Assigned 2 worker(s)", res)
        self.assertEqual(self.env.state["workers"].get("farm", 0), 2)
        self.assertEqual(self.env.state["population"]["idle"], 3)

    def test_advance_turn_production_and_consumption(self):
        self.env.step("build", {"building_type": "farm"})
        self.env.step("assign_workers", {"building_type": "farm", "count": 2})
        food_before = self.env.state["resources"]["food"]
        total_pop = self.env.state["population"]["total"]

        # Advance turn: farm produces 2*4=8 food. Pop consumes total_pop (5) food. Net +3 food.
        res, done, info = self.env.step("advance_turn", {})
        self.assertIn("Turn 1 -> 2", res)
        food_after = self.env.state["resources"]["food"]
        self.assertEqual(food_after, food_before + (2 * 4) - total_pop)

    def test_trade_market_required(self):
        # Without market, trading fails
        res, done, info = self.env.step("trade", {"action": "sell", "resource": "wood", "amount": 5})
        self.assertIn("You need a market to trade", res)

    def test_step_limit(self):
        # Step limit is 40
        for _ in range(40):
            if self.env.done:
                break
            self.env.step("view_kingdom", {})
        self.assertTrue(self.env.done)
        self.assertEqual(self.env.termination_reason, "step_cap")


if __name__ == "__main__":
    unittest.main()
