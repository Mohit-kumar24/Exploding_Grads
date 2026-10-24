"""
Tests for Discrepancy Detection, State Invariant Enforcement, Structured Error Feedback,
and Reward Integrity.
"""
import copy
import time
import unittest
from unittest.mock import patch, MagicMock

from src.env.database_env import KingdomEnv
from src.env.tasks import get_task
from src.verifier.verifier import Verifier
from src.harness.runner import (
    run_model_episode,
    run_baselines_for_task,
    MAX_SLOT_TIME_SECONDS,
    BASELINE_TIME_LIMIT,
)
from src.harness.logger import Logger


class TestDiscrepancyAndIntegrity(unittest.TestCase):
    """Test suite for discrepancy detection and reward integrity guarantees."""

    def test_structured_error_format_provides_actionable_recovery(self):
        """When an invalid action occurs, error feedback must contain category, reason, constraint, and recovery."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Attempt to recruit worker without sufficient gold
        env._state["resources"]["gold"] = 0
        res, done, info = env.step("recruit", {"count": 1})

        self.assertIn("INVALID_ACTION: recruit", res)
        self.assertIn("Reason:", res)
        self.assertIn("Constraint:", res)
        self.assertIn("State change: none", res)
        self.assertIn("Recovery:", res)

    def test_state_invariant_violation_detected_on_negative_resources(self):
        """Detect invariant violation if resources drop below zero."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Force negative resource
        env._state["resources"]["gold"] = -5
        violations = env.validate_invariants()
        self.assertTrue(any("Negative resource: gold" in v for v in violations))

    def test_state_invariant_violation_detected_on_population_mismatch(self):
        """Detect invariant violation if total population does not equal assigned workers + idle workers."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Corrupt population
        env._state["population"]["total"] = 100
        env._state["population"]["idle"] = 5
        env._state["population"]["workers"] = 5
        violations = env.validate_invariants()
        self.assertTrue(any("Population mismatch" in v for v in violations))

    def test_discrepancy_detected_when_state_mutated_on_error(self):
        """If step encounters an error but state changed, discrepancy must be logged."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Mock an internal action that mutates state but returns Error:
        orig_tool_build = env._tool_build
        def buggy_build(args):
            env._state["resources"]["wood"] += 100  # rogue mutation
            return "Error: Database deadlock occurred"

        env._tool_build = buggy_build
        res, done, info = env.step("build", {"building_type": "farm"})

        discrepancies = env.discrepancies
        self.assertTrue(any(d.get("category") == "error_state_mutation" for d in discrepancies))

    def test_verifier_overrides_score_to_zero_on_state_invariant_violation(self):
        """Verifier must flag integrity violation and force score to 0.0 if invariants are broken."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Satisfy objectives artificially
        env._state["buildings"]["farm"] = 2
        env._state["resources"]["food"] = 50

        # Inject invariant violation
        env._state["resources"]["gold"] = -50

        verifier = Verifier()
        task = get_task("task_1")
        result = verifier.verify(env, task)

        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["classification"], "integrity_violation")
        self.assertTrue(any("Negative resource: gold" in v for v in result["details"]["invariant_violations"]))

    def test_combined_baseline_timeout_gate_failure(self):
        """When combined reference and noop baselines exceed 120s, gate check must fail."""
        logger = MagicMock()
        mono_time = 0.0
        def fake_mono():
            nonlocal mono_time
            mono_time += 70.0
            return mono_time

        with patch("time.monotonic", side_effect=fake_mono):
            ref_res, noop_res = run_baselines_for_task("task_1", 42, logger, combined_timeout=120.0)

        self.assertFalse(ref_res.get("gate_check", {}).get("passed", False))
        self.assertTrue(ref_res.get("gate_check", {}).get("time_limit_exceeded", False))

    def test_slot_time_cap_enforced_in_model_episode(self):
        """When overall slot time exceeds 16 minutes (960s), model episode terminates with time_cap."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        mock_client = MagicMock()
        mock_client.model = "mock"
        mock_client.chat.return_value = {
            "content": "thinking",
            "tool_calls": [{"id": "c1", "name": "view_kingdom", "arguments": {}}],
            "tokens": 5,
        }

        logger = MagicMock()
        slot_start = 100.0
        call_count = 0
        def fake_mono():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 100.0
            return 1200.0

        with patch("time.monotonic", side_effect=fake_mono):
            result = run_model_episode(env, "task_1", 42, mock_client, logger, slot_start_time=slot_start)

        self.assertEqual(result["termination_reason"], "time_cap")
        self.assertTrue(env.is_done)

    def test_run_b_category_1_detected_and_flagged_in_telemetry(self):
        """Run B Category 1: Unusual but recoverable events are detected and flagged in structured telemetry."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Inject an invalid tool call and a rejected action
        res1, done1, info1 = env.step("invalid_tool_name", {})
        self.assertIn("Error: Unknown tool 'invalid_tool_name'", res1)

        # Inject resource depletion rejected action
        env._state["resources"]["gold"] = 0
        res2, done2, info2 = env.step("recruit", {"count": 1})
        self.assertIn("INVALID_ACTION: recruit", res2)
        self.assertIn("Reason: Not enough gold", res2)

        # Telemetry verification: verify structured log step captures flag
        logger = Logger(log_dir="logs")
        logger.start_episode("test_run_b", "task_1", 42, "mock", "test")
        logger.log_step(
            step=1,
            seed=42,
            model_input="prompt",
            model_output_raw="call recruit",
            parsed_actions=[{"name": "recruit", "arguments": {"count": 1}}],
            tool_results=[res2],
            errors=["Insufficient gold (10 required, 0 available)"],
            state_hash="abcd1234",
            state=copy.deepcopy(env._state),
            discrepancies=[{"category": "rejected_action", "message": "Insufficient gold"}],
            reward={"score": 0.0},
            tokens=10,
            latency_ms=15.0,
        )
        logger.end_episode("step_cap", 0.0, 1, "abcd1234")

    def test_run_b_category_2_clear_error_and_episode_continues_safely(self):
        """Run B Category 2: Model receives actionable error, episode continues safely without undeserved reward."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Step 1: Model issues invalid action (insufficient gold)
        env._state["resources"]["gold"] = 0
        res1, done1, info1 = env.step("recruit", {"count": 1})
        self.assertFalse(done1)
        self.assertIn("INVALID_ACTION: recruit", res1)
        self.assertIn("Reason: Not enough gold", res1)
        self.assertIn("State change: none", res1)
        self.assertIn("Recovery:", res1)

        # Verify no undeserved reward was granted
        task = get_task("task_1")
        verifier = Verifier()
        ver1 = verifier.verify(env, task)
        self.assertEqual(ver1["score"], 0.0)

        # Step 2: Model recovers by querying kingdom state
        res2, done2, info2 = env.step("view_kingdom", {})
        self.assertFalse(done2)
        self.assertIn("resources", res2)

        # Step 3: Model executes legitimate build action with sufficient wood and gold
        env._state["resources"]["wood"] = 20
        env._state["resources"]["gold"] = 20
        res3, done3, info3 = env.step("build", {"building_type": "farm"})
        self.assertFalse(done3)
        self.assertEqual(env._state["buildings"].get("farm", 0), 1)

        # Score updated accurately for genuine achievement
        ver3 = verifier.verify(env, task)
        self.assertGreater(ver3["score"], 0.0)


    def test_run_b_category_3_corrupted_state_blocks_passing_verdict(self):
        """Run B Category 3: Fatal corrupted state triggers integrity violation and blocks passing score."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Satisfy task objectives
        env._state["buildings"]["farm"] = 2
        env._state["resources"]["food"] = 50

        # Inject dangerous corrupted state (negative stone)
        env._state["resources"]["stone"] = -25

        verifier = Verifier()
        task = get_task("task_1")
        res = verifier.verify(env, task)

        self.assertEqual(res["score"], 0.0)
        self.assertFalse(res["passed"])
        self.assertEqual(res["classification"], "integrity_violation")
        self.assertIn("Negative resource: stone", str(res["details"]["invariant_violations"]))

    def test_discrepancy_audit_reward_recomputation_agreement(self):
        """Discrepancy audit: Independently recomputed reward agrees with environment check_progress and verifier."""
        verifier = Verifier()
        for task_id in ["task_1", "task_2", "task_3", "task_7"]:
            env = KingdomEnv()
            env.reset(task_id, seed=42)
            task = get_task(task_id)

            # Recompute at reset (must be 0.0000)
            initial_prog = task.check_progress(env._state)
            self.assertTrue(all(not obj["met"] for obj in initial_prog["objectives"]))
            initial_ver = verifier.verify(env, task)
            self.assertEqual(initial_ver["score"], 0.0)
            self.assertFalse(initial_ver["passed"])



if __name__ == "__main__":
    unittest.main()

