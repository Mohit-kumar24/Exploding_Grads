"""
Tests for Execution Limits, Boundaries, and Timeouts.

Verifies:
1. Per-tool-call timeout: stalled tool calls abort with timeout error within limit.
2. Max tool calls per reply: calls exceeding MAX_TOOL_CALLS_PER_REPLY (10) are rejected.
3. Max steps: episodes terminate strictly at MAX_STEPS (40) with reason='step_cap'.
4. Overall episode duration: exceeding MAX_TIME_SECONDS terminates with reason='time_cap'.
5. Combined baseline execution limit: baseline exceeding 120s fails gate check.
"""
import time
import unittest
from unittest.mock import patch, MagicMock

from src.env.database_env import KingdomEnv
from src.harness.runner import (
    execute_tool_call,
    run_model_episode,
    MAX_STEPS,
    MAX_TOOL_CALLS_PER_REPLY,
    MAX_TIME_SECONDS,
    TOOL_CALL_TIMEOUT,
    BASELINE_TIME_LIMIT,
)
from src.harness.logger import Logger
from src.verifier.verifier import Verifier


class MockStalledEnv(KingdomEnv):
    """Mock environment that sleeps to simulate a hanging tool call and attempts rogue mutations."""
    def __init__(self):
        super().__init__()
        self.reset("task_1", seed=42)

    def step(self, tool_name, args):
        time.sleep(0.4)
        # Attempt rogue state mutation on whatever env object it has
        self._state["resources"]["gold"] = 999999
        return "finished", False, {}


class TestExecutionLimits(unittest.TestCase):
    """Test suite for hard limit enforcement."""

    def test_stalled_tool_call_times_out_and_does_not_hang(self):
        """A stalled tool call must abort at timeout without waiting for stalled thread."""
        env = MockStalledEnv()
        start = time.time()
        # Test with a tight timeout of 0.05 seconds
        result_str, done, info = execute_tool_call(env, "view_kingdom", {}, timeout=0.05)
        elapsed = time.time() - start

        self.assertIn("timed out", result_str)
        self.assertEqual(info.get("error"), "timeout")
        # Elapsed must be well under the 0.4s stall duration (proves non-blocking shutdown)
        self.assertLess(elapsed, 0.25, f"Harness waited for stalled worker: {elapsed:.3f}s >= 0.25s")

    def test_stalled_tool_cannot_mutate_live_environment_state(self):
        """A stalled tool that mutates state after timeout must NOT corrupt live environment."""
        env = MockStalledEnv()
        self.assertEqual(env.state["resources"]["gold"], 30)

        # Execute with 0.05s timeout while step takes 0.4s and sets gold=999999
        execute_tool_call(env, "view_kingdom", {}, timeout=0.05)

        # Wait for the rogue worker to finish its sleep in background
        time.sleep(0.45)

        # Live environment must remain uncorrupted (initial gold=30, NOT 999999)
        self.assertEqual(env.state["resources"]["gold"], 30, "Rogue timed-out tool mutated live state!")

    def test_stalled_baseline_aborts_during_execution_and_fails_gate(self):
        """A stalled baseline agent must be aborted actively during execution, not after."""
        from src.harness.runner import run_baseline
        env = KingdomEnv()
        verifier = Verifier()
        logger = MagicMock()

        class HangingAgent:
            def run_episode(self, env, task_id, seed=42):
                time.sleep(0.5)
                return {"steps": 1, "trajectory": []}

        start = time.time()
        with patch("src.harness.runner.ReferenceAgent", return_value=HangingAgent()):
            result = run_baseline(env, "task_1", 42, "reference", verifier, logger, timeout=0.05)
        elapsed = time.time() - start

        self.assertFalse(result["gate_check"]["passed"])
        self.assertTrue(result["gate_check"].get("time_limit_exceeded"))
        self.assertLess(elapsed, 0.3, f"Baseline runner waited for stalled agent: {elapsed:.3f}s >= 0.3s")

    def test_max_tool_calls_per_reply_enforced(self):
        """Tool calls beyond 10 per reply must be rejected."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Mock model returning 14 tool calls
        calls = [{"id": f"call_{i}", "name": "view_kingdom", "arguments": {}} for i in range(14)]
        mock_client = MagicMock()
        mock_client.model = "mock"
        mock_client.chat.side_effect = [
            {"content": "reply 1", "tool_calls": calls, "tokens": 10},
            {"content": "done", "tool_calls": [{"id": "sub", "name": "submit_answer", "arguments": {"answer": "done"}}], "tokens": 10},
        ]

        logger = MagicMock()
        with patch("src.harness.runner._load_append_safe", return_value=""):
            result = run_model_episode(env, "task_1", 42, mock_client, logger)

        # Find logger.log_step calls
        step_calls = [c for c in logger.log_step.call_args_list]
        self.assertGreater(len(step_calls), 0)

        # In first step, actions 10-13 must have been rejected
        first_step_actions = step_calls[0].kwargs.get("parsed_actions", [])
        self.assertEqual(len(first_step_actions), 14)
        for act in first_step_actions[10:]:
            self.assertTrue(act.get("rejected"))

    def test_max_episode_steps_enforced(self):
        """Episode must terminate strictly at 40 steps with step_cap reason."""
        env = KingdomEnv()
        env.reset("task_1", seed=42)

        # Model that does useless actions endlessly
        mock_client = MagicMock()
        mock_client.model = "mock"
        mock_client.chat.return_value = {
            "content": "thinking",
            "tool_calls": [{"id": "c1", "name": "view_kingdom", "arguments": {}}],
            "tokens": 5,
        }

        logger = MagicMock()
        result = run_model_episode(env, "task_1", 42, mock_client, logger)

        self.assertEqual(result["steps_used"], MAX_STEPS)
        self.assertEqual(result["termination_reason"], "step_cap")
        self.assertTrue(env.is_done)

    def test_overall_episode_duration_enforced(self):
        """When total elapsed time exceeds MAX_TIME_SECONDS, episode ends with time_cap."""
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
        curr_time = 1000.0
        def fake_time():
            nonlocal curr_time
            curr_time += 650.0  # advances beyond MAX_TIME_SECONDS (600s)
            return curr_time

        with patch("time.monotonic", side_effect=fake_time), patch("time.time", side_effect=fake_time):
            result = run_model_episode(env, "task_1", 42, mock_client, logger)

        self.assertEqual(result["termination_reason"], "time_cap")
        self.assertTrue(env.is_done)


if __name__ == "__main__":
    unittest.main()
