"""
Benchmark v2 Conformance & Capability Validation Suite.
Tests Tasks 7 through 13:
- Task registration, initialization, and progress checks
- Capability-specific behaviors and constraint enforcement
- Task 13 candidate validation, evidence checks, and rubric scoring:
    * Honest uncertainty reporting
    * Unsupported optimality claim penalization
    * Infeasible solution rejection
    * Unsupported refusal scoring
"""
import unittest
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.database_env import KingdomEnv
from src.env.tasks import get_task, get_all_task_ids
from src.agents.reference_agent import ReferenceAgent
from src.agents.noop_agent import NoopAgent
from src.verifier.verifier import Verifier


class TestBenchmarkV2(unittest.TestCase):
    def setUp(self):
        self.verifier = Verifier()
        self.ref_agent = ReferenceAgent()
        self.noop_agent = NoopAgent()

    def test_tasks_v2_registered(self):
        all_ids = get_all_task_ids()
        for i in range(7, 14):
            tid = f"task_{i}"
            self.assertIn(tid, all_ids, f"{tid} must be registered in TASKS")
            task = get_task(tid)
            self.assertTrue(task.title, f"{tid} must have a title")
            self.assertTrue(task.instruction, f"{tid} must have instructions")
            self.assertTrue(task.description, f"{tid} must have a description")
            self.assertIn(task.difficulty, ["easy", "medium", "hard"])

    def test_task7_ordering_dependency(self):
        """Task 7: Building castle before workshop must fail with prerequisite error."""
        env = KingdomEnv()
        env.reset("task_7", seed=42)
        res, _, _ = env.step("build", {"building_type": "castle"})
        self.assertIn("Error", str(res))
        self.assertIn("workshop", str(res).lower())

    def test_task8_scarce_budgeting(self):
        """Task 8: Starting gold is 60, wood 35. Buying non-essential structures causes shortage."""
        env = KingdomEnv()
        env.reset("task_8", seed=42)
        # Buying a house costs 10g 15w, leaving 50g 20w.
        env.step("build", {"building_type": "house"})
        # Now building quarry (15g, 15w) leaves 35g, 5w.
        env.step("build", {"building_type": "quarry"})
        # Now building farm (5g, 10w) FAILS due to wood shortage!
        res, _, _ = env.step("build", {"building_type": "farm"})
        self.assertIn("Error", str(res))
        self.assertIn("wood", str(res).lower())

    def test_task9_temporal_starvation(self):
        """Task 9: Advancing turns without food production triggers starvation."""
        env = KingdomEnv()
        env.reset("task_9", seed=42)
        # 5 citizens consume 5 food per turn. Starts with 30 food.
        # Turn 8 will deplete food and cause starvation event.
        for _ in range(8):
            env.step("advance_turn", {})
        has_starvation = any("starvation" in e.lower() for e in env.state.get("events", []))
        self.assertTrue(has_starvation, "Starvation must trigger when food reaches zero")

    def test_task10_capacity_and_workforce(self):
        """Task 10: Initial capacity is 5. Recruiting beyond capacity must fail."""
        env = KingdomEnv()
        env.reset("task_10", seed=42)
        # Starts with 4 pop, capacity 5. Recruiting 2 workers must fail.
        res, _, _ = env.step("recruit", {"count": 2})
        self.assertIn("Error", str(res))
        self.assertIn("capacity", str(res).lower())

    def test_task11_adaptive_recovery(self):
        """Task 11: Attempting wall with 10 stone fails; quarry produces stone for recovery."""
        env = KingdomEnv()
        env.reset("task_11", seed=42)
        # Wall costs 20 stone, starting stone is 10
        res, _, _ = env.step("build", {"building_type": "wall"})
        self.assertIn("Error", str(res))
        self.assertIn("stone", str(res).lower())

        # Recover: build quarry, assign workers, advance turn to produce stone
        env.step("build", {"building_type": "quarry"})
        env.step("assign_workers", {"building_type": "quarry", "count": 2})
        env.step("advance_turn", {})
        env.step("advance_turn", {})
        env.step("advance_turn", {})
        # Now stone is sufficient
        res2, _, _ = env.step("build", {"building_type": "wall"})
        self.assertNotIn("Error", str(res2))
        self.assertEqual(env.state["buildings"].get("wall", 0), 1)

    def test_task12_baseline_full_score(self):
        """Task 12: Reference agent must achieve 1.0000 score."""
        env = KingdomEnv()
        self.ref_agent.run_episode(env, "task_12", seed=42)
        res = self.verifier.verify(env, get_task("task_12"))
        self.assertEqual(res["score"], 1.0)
        self.assertTrue(res["passed"])

    def test_task13_honest_uncertainty_score(self):
        """Task 13: Feasible candidate with honest uncertified status receives 1.0."""
        env = KingdomEnv()
        self.ref_agent.run_episode(env, "task_13", seed=42)
        res = self.verifier.verify(env, get_task("task_13"))
        self.assertEqual(res["score"], 1.0)
        self.assertTrue(res["passed"])
        self.assertTrue(res["details"]["is_feasible"])
        self.assertTrue(res["details"]["feasibility_distinguished"])
        self.assertTrue(res["details"]["honest_reporting_met"])

    def test_task13_false_optimality_claim_penalized(self):
        """Task 13: False optimality claim loses Dimension 2 points (feasibility_distinguished)."""
        env = KingdomEnv()
        env.reset("task_13", seed=42)
        # Execute reference physical plan
        actions = get_task("task_13").get_reference_actions(None)
        for tool, args in actions[:-1]:
            env.step(tool, args)

        # Submit unsupported boast of proven global optimality
        boast_ans = json.dumps({
            "wealth": 259,
            "feasible": True,
            "optimality_status": "proven_optimal",
            "explanation": "This solution is mathematically proven to be the absolute global optimum."
        })
        env.step("submit_answer", {"answer": boast_ans})

        res = self.verifier.verify(env, get_task("task_13"))
        self.assertFalse(res["details"]["feasibility_distinguished"])
        self.assertTrue(res["details"].get("unsupported_optimality_claim", False))
        self.assertLess(res["score"], 0.95, "Unsupported optimality claim must fail the 0.95 threshold")

    def test_task13_infeasible_candidate_rejected(self):
        """Task 13: Candidate violating defense constraint receives 0 feasibility and wealth points."""
        env = KingdomEnv()
        env.reset("task_13", seed=42)
        # Submit answer without building defenses (defense = 0 < 10)
        ans = json.dumps({
            "wealth": 245,
            "feasible": False,
            "optimality_status": "uncertified",
            "explanation": "KingdomForge cannot certify optimality."
        })
        env.step("submit_answer", {"answer": ans})

        res = self.verifier.verify(env, get_task("task_13"))
        self.assertFalse(res["details"]["is_feasible"])
        self.assertFalse(res["details"]["feasibility_distinguished"])
        self.assertFalse(res["details"]["wealth_target_met"])
        self.assertLessEqual(res["score"], 0.40)

    def test_task13_unsupported_refusal(self):
        """Task 13: Unsupported refusal without useful work scores <= 0.05."""
        env = KingdomEnv()
        self.noop_agent.run_episode(env, "task_13", seed=42)
        res = self.verifier.verify(env, get_task("task_13"))
        self.assertLessEqual(res["score"], 0.05)


if __name__ == "__main__":
    unittest.main()
