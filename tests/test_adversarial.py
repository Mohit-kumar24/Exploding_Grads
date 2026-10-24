"""
Targeted Adversarial Robustness and Exploit Verification Test Suite for KingdomForge.

Validates:
1. Starvation Mechanics: Confirms starving population is deducted from idle first,
   then from assigned workers, strictly preserving total == idle + sum(workers.values()).
2. Task Difficulty & Trivial Solvability: Evaluates Task 2's direct solvability on Turn 1
   against its exact instruction criteria.
3. Parameter Type Enforcement: Confirms boolean inputs (True/False) are strictly rejected
   from integer parameters across assign_workers, unassign_workers, trade, and recruit.
4. Economic Arbitrage: Confirms circular resource conversion produces net negative returns.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.database_env import KingdomEnv
from src.env.tasks import get_task
from src.verifier.verifier import Verifier


class TestAdversarialAttacks(unittest.TestCase):
    def setUp(self):
        self.verifier = Verifier()

    def test_starvation_preserves_population_consistency_and_kills_assigned_workers(self):
        """
        Confirms starvation deducts from assigned workers when idle is exhausted
        and guarantees total == idle + sum(workers.values()) at all times.
        """
        env = KingdomEnv()
        env.reset("task_2", seed=42)
        env.step("build", {"building_type": "lumber_mill"})
        env.step("assign_workers", {"building_type": "lumber_mill", "count": 3})

        # State: 3 assigned to lumber_mill, 2 idle, total 5, food 0
        env._state["population"]["idle"] = 1
        env._state["population"]["total"] = 4
        env._state["resources"]["food"] = 0

        # Advance turn: Lumber mill produces 3 * 3 = 9 wood, 0 food.
        # Population (4) consumes 4 food -> food drops to -4. Deficit = 4.
        msg, done, info = env.step("advance_turn", {})

        # Deficit of 4 with 1 idle and 3 assigned:
        # 1 idle must die, all 3 assigned must die -> total deaths = 4 -> total pop = 0
        self.assertEqual(env.state["population"]["total"], 0)
        self.assertEqual(env.state["population"]["idle"], 0)
        self.assertEqual(sum(env.state["workers"].values()), 0)
        self.assertEqual(env.state["resources"]["food"], 0)
        # Mathematical invariant holds
        self.assertEqual(env.state["population"]["total"],
                         env.state["population"]["idle"] + sum(env.state["workers"].values()))
        self.assertIn("STARVATION: 4 people left the kingdom", msg)

    def test_starvation_partial_assigned_deduction(self):
        """
        Confirms that when deficit partially exceeds idle, only the required number
        of assigned workers are removed, preserving surviving workers.
        """
        env = KingdomEnv()
        env.reset("task_2", seed=42)
        env.step("build", {"building_type": "lumber_mill"})
        env.step("assign_workers", {"building_type": "lumber_mill", "count": 3})

        # 3 assigned, 1 idle, total 4
        env._state["population"]["idle"] = 1
        env._state["population"]["total"] = 4
        # Give 2 food, so 4 people consume 4 food -> deficit is -2.
        env._state["resources"]["food"] = 2

        msg, done, info = env.step("advance_turn", {})

        # Deficit = 2. 1 idle dies, 1 assigned worker dies.
        # Remaining: 0 idle, 2 assigned to lumber_mill, total 2.
        self.assertEqual(env.state["population"]["total"], 2)
        self.assertEqual(env.state["population"]["idle"], 0)
        self.assertEqual(env.state["workers"].get("lumber_mill", 0), 2)
        # Invariant check
        self.assertEqual(env.state["population"]["total"],
                         env.state["population"]["idle"] + sum(env.state["workers"].values()))
        self.assertIn("STARVATION: 2 people left the kingdom", msg)

    def test_task2_trivial_solvability_documented(self):
        """
        Evaluates Task 2 difficulty: confirms that according to the exact instruction:
        'Build the basic infrastructure for your kingdom: construct a farm, a lumber mill,
         and a house. Grow your population to at least 6 people... capacity >= 10',
        the task can be solved on Turn 1 in 5 steps because starting resources exceed costs.
        """
        env = KingdomEnv()
        env.reset("task_2", seed=42)
        env.step("build", {"building_type": "farm"})
        env.step("build", {"building_type": "lumber_mill"})
        env.step("build", {"building_type": "house"})
        env.step("recruit", {"count": 1})
        env.step("submit_answer", {"answer": "done"})

        task = get_task("task_2")
        result = self.verifier.verify(env, task)

        # Proves Task 2 is trivially solvable under its specification
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["passed"])
        self.assertEqual(env._turn, 1)

    def test_boolean_type_confusion_strictly_rejected(self):
        """
        Adversarial Test: Agent passes boolean True to integer parameters across all tools.
        Must be strictly rejected with integer validation error.
        """
        env = KingdomEnv()
        env.reset("task_1", seed=42)
        env.step("build", {"building_type": "farm"})

        res1, _, _ = env.step("assign_workers", {"building_type": "farm", "count": True})
        self.assertIn("Error: count must be a positive integer.", res1)

        res2, _, _ = env.step("unassign_workers", {"building_type": "farm", "count": True})
        self.assertIn("Error: count must be a positive integer.", res2)

        res3, _, _ = env.step("recruit", {"count": True})
        self.assertIn("Error: count must be a positive integer.", res3)

        env._state["buildings"]["market"] = 1
        res4, _, _ = env.step("trade", {"action": "sell", "resource": "food", "amount": True})
        self.assertIn("Error: amount must be a positive integer.", res4)

    def test_circular_trade_arbitrage_prevented(self):
        """
        Adversarial Test: Check that trading food -> gold -> food yields a net loss.
        """
        env = KingdomEnv()
        env.reset("task_3", seed=42)
        env._state["buildings"]["market"] = 1
        env._state["resources"]["food"] = 30
        env._state["resources"]["gold"] = 0

        # Sell 10 food -> gains 20 gold
        env.step("trade", {"action": "sell", "resource": "food", "amount": 10})
        gold = env.state["resources"]["gold"]
        self.assertEqual(gold, 20)

        # Buy food with 20 gold -> costs 3g per food -> can buy at most 6 food (costs 18g)
        env.step("trade", {"action": "buy", "resource": "food", "amount": 6})
        self.assertEqual(env.state["resources"]["food"], 26)  # Started with 30, sold 10, bought 6 = 26 (net loss of 4)


if __name__ == "__main__":
    unittest.main()
