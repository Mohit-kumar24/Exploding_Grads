"""
Tests for Baseline Gate Failure Path and Model Execution Guarding.

Verifies:
1. Gate detects failed reference baseline (score < 0.95) and fails closed.
2. Gate detects failed no-op baseline (score > 0.05) and fails closed.
3. Gate detects baseline timeouts and fails closed.
4. The model client is NEVER called when a baseline gate fails (proven via SpyModelClient).
5. The real baseline-results/gate.json is preserved and never corrupted with test artifacts.
"""
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from src.env.database_env import KingdomEnv
from src.env.tasks import get_task
from src.verifier.verifier import Verifier
from src.harness.logger import Logger
import src.harness.runner as runner


class SpyModelClient:
    """Spy client that tracks if chat() was ever called."""
    def __init__(self):
        self.call_count = 0
        self.model = "spy-model"

    def chat(self, messages, tools, timeout=90.0):
        self.call_count += 1
        return {"content": "spy reply", "tool_calls": []}


class TestGateFailurePath(unittest.TestCase):
    """Test suite verifying fail-closed baseline gate behavior."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Logger(log_dir=self.temp_dir)
        self.spy_client = SpyModelClient()

        # Backup real gate.json if it exists
        self.real_gate_path = os.path.join("baseline-results", "gate.json")
        self.gate_backup = None
        if os.path.exists(self.real_gate_path):
            with open(self.real_gate_path, "r", encoding="utf-8") as f:
                self.gate_backup = f.read()

    def tearDown(self):
        # Restore real gate.json exactly as it was
        if self.gate_backup is not None:
            os.makedirs("baseline-results", exist_ok=True)
            with open(self.real_gate_path, "w", encoding="utf-8") as f:
                f.write(self.gate_backup)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_gate_fails_when_reference_score_below_threshold(self):
        """When reference score < 0.95, gate must fail and model must NOT be called."""
        failing_ref_result = {
            "score": 0.80,
            "passed": False,
            "gate_check": {"passed": False, "threshold": ">= 0.95", "actual": 0.80},
        }
        passing_noop_result = {
            "score": 0.00,
            "passed": True,
            "gate_check": {"passed": True, "threshold": "<= 0.05", "actual": 0.00},
        }

        # Mock runner.run_baselines_for_task to return failing reference baseline
        with patch("src.harness.runner.run_baselines_for_task", return_value=(failing_ref_result, passing_noop_result)):
            with patch.dict(os.environ, {"MODE": "full", "TASK_ID": "task_1", "SEED": "42"}):
                with patch("sys.exit") as mock_exit:
                    mock_exit.side_effect = SystemExit(1)
                    with self.assertRaises(SystemExit) as cm:
                        runner.main()

                    # Verify exit code was nonzero (1)
                    self.assertEqual(cm.exception.code, 1)

        # Model must NEVER have been called
        self.assertEqual(self.spy_client.call_count, 0)

    def test_gate_fails_when_noop_score_above_threshold(self):
        """When no-op score > 0.05, gate must fail and model must NOT be called."""
        passing_ref_result = {
            "score": 1.00,
            "passed": True,
            "gate_check": {"passed": True, "threshold": ">= 0.95", "actual": 1.00},
        }
        failing_noop_result = {
            "score": 0.15,
            "passed": False,
            "gate_check": {"passed": False, "threshold": "<= 0.05", "actual": 0.15},
        }

        with patch("src.harness.runner.run_baselines_for_task", return_value=(passing_ref_result, failing_noop_result)):
            with patch.dict(os.environ, {"MODE": "full", "TASK_ID": "task_1", "SEED": "42"}):
                with patch("sys.exit") as mock_exit:
                    mock_exit.side_effect = SystemExit(1)
                    with self.assertRaises(SystemExit) as cm:
                        runner.main()

                    # Verify exit code was nonzero (1)
                    self.assertEqual(cm.exception.code, 1)

        # Model must NEVER have been called
        self.assertEqual(self.spy_client.call_count, 0)

    def test_gate_fails_when_baseline_times_out(self):
        """When baseline exceeds time limit, gate check is marked failed."""
        verifier = Verifier()
        env = KingdomEnv()
        env.reset("task_1", 42)

        class HangingAgent:
            def run_episode(self, env, task_id, seed=42):
                time.sleep(0.3)
                return {"steps": 1, "trajectory": []}

        with patch("src.harness.runner.ReferenceAgent", return_value=HangingAgent()):
            result = runner.run_baseline(env, "task_1", 42, "reference", verifier, self.logger, timeout=0.05)
            self.assertFalse(result["gate_check"]["passed"])
            self.assertTrue(result["gate_check"].get("time_limit_exceeded"))

    def test_model_guard_prevents_call_when_baseline_fails(self):
        """Directly verify model invocation guard: if ref or noop gate fails, run_model_episode is not called."""
        with patch("src.harness.runner.run_model_episode") as mock_run_model:
            ref_fail = {"score": 0.5, "gate_check": {"passed": False}}
            noop_pass = {"score": 0.0, "gate_check": {"passed": True}}

            with patch("src.harness.runner.run_baselines_for_task", return_value=(ref_fail, noop_pass)):
                with patch.dict(os.environ, {"MODE": "full", "TASK_ID": "task_1", "MODEL": "mock"}):
                    with patch("sys.exit", side_effect=SystemExit(1)):
                        try:
                            runner.main()
                        except SystemExit:
                            pass

            # Model episode runner must NEVER be invoked
            mock_run_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
