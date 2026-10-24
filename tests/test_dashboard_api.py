"""
Unit tests for KingdomForge Dashboard Data & API

Tests all 10 verification scenarios required by specification:
1. Normal populated results directory
2. A new run appearing after dashboard starts
3. Empty or missing results files
4. Malformed JSON/JSONL records
5. An incomplete or interrupted evaluation
6. A model/API crash (distinguished from planning failure)
7. Multiple tasks and multiple seeds
8. A changed task inventory
9. Missing optional metrics
10. Attempts to access paths outside approved result directories (path traversal)
"""
import json
import os
import shutil
import tempfile
import unittest

from src.harness.dashboard_data import (
    compute_summary,
    get_baseline_gates,
    get_run_trajectory,
    get_task_inventory,
    sanitize_filename,
    scan_run_logs,
)


class TestDashboardData(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_dashboard_")
        self.logs_dir = os.path.join(self.test_dir, "logs")
        self.gate_dir = os.path.join(self.test_dir, "baseline-results")
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.gate_dir, exist_ok=True)
        self.gate_path = os.path.join(self.gate_dir, "gate.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_run(self, filename, lines):
        path = os.path.join(self.logs_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return path

    # 1. Normal populated results directory
    def test_01_normal_populated_results(self):
        self._write_run("model_task_1_run1.jsonl", [
            {"type": "episode_start", "timestamp": "2026-10-24T00:00:00Z", "task_id": "task_1", "seed": 42, "model": "test-model", "mode": "model"},
            {"type": "step", "step": 1, "parsed_actions": [{"tool": "view_kingdom", "args": {}}], "tool_result": ["ok"], "state": {"hash": "abc"}},
            {"type": "verification", "score": 1.0, "passed": True, "details": {}},
            {"type": "episode_end", "termination_reason": "submitted", "final_score": 1.0, "total_steps": 1, "final_state_hash": "abc"}
        ])
        runs = scan_run_logs(log_dirs=[self.logs_dir])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["task_id"], "task_1")
        self.assertEqual(runs[0]["final_score"], 1.0)
        self.assertTrue(runs[0]["passed"])
        self.assertTrue(runs[0]["is_completed"])
        self.assertFalse(runs[0]["is_crash"])

    # 2. A new run appearing after dashboard starts
    def test_02_new_run_appearing_dynamically(self):
        runs_before = scan_run_logs(log_dirs=[self.logs_dir])
        self.assertEqual(len(runs_before), 0)

        # New run created
        self._write_run("model_task_2_run2.jsonl", [
            {"type": "episode_start", "timestamp": "2026-10-24T01:00:00Z", "task_id": "task_2", "seed": 42, "model": "test-model", "mode": "model"},
            {"type": "verification", "score": 0.8, "passed": False, "details": {}},
            {"type": "episode_end", "termination_reason": "step_cap", "final_score": 0.8, "total_steps": 40, "final_state_hash": "def"}
        ])

        runs_after = scan_run_logs(log_dirs=[self.logs_dir])
        self.assertEqual(len(runs_after), 1)
        self.assertEqual(runs_after[0]["task_id"], "task_2")
        self.assertEqual(runs_after[0]["final_score"], 0.8)

    # 3. Empty or missing results files
    def test_03_empty_or_missing_results(self):
        # Empty file
        empty_path = os.path.join(self.logs_dir, "empty_run.jsonl")
        open(empty_path, "w").close()

        runs = scan_run_logs(log_dirs=[self.logs_dir])
        self.assertEqual(len(runs), 0)

        # Missing gate.json
        missing_gates = get_baseline_gates(os.path.join(self.test_dir, "nonexistent.json"))
        self.assertEqual(missing_gates, {})

    # 4. Malformed JSON/JSONL records
    def test_04_malformed_json_records(self):
        path = os.path.join(self.logs_dir, "corrupt_run.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"type": "episode_start", "task_id": "task_1", "mode": "model"}\n')
            f.write('NOT_VALID_JSON_LINE\n')
            f.write('{"type": "episode_end", "termination_reason": "submitted", "final_score": 0.5}\n')

        runs = scan_run_logs(log_dirs=[self.logs_dir])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["final_score"], 0.5)

    # 5. An incomplete or interrupted evaluation
    def test_05_incomplete_interrupted_evaluation(self):
        self._write_run("interrupted_run.jsonl", [
            {"type": "episode_start", "timestamp": "2026-10-24T02:00:00Z", "task_id": "task_3", "seed": 42, "model": "test-model", "mode": "model"},
            {"type": "step", "step": 1, "parsed_actions": [{"tool": "view_kingdom", "args": {}}], "tool_result": ["ok"]}
            # No episode_end!
        ])
        runs = scan_run_logs(log_dirs=[self.logs_dir])
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0]["is_interrupted"])
        self.assertFalse(runs[0]["is_completed"])
        self.assertEqual(runs[0]["termination_reason"], "in_progress")

    # 6. A model/API crash (distinguished from planning failure)
    def test_06_model_api_crash_distinction(self):
        self._write_run("crashed_run.jsonl", [
            {"type": "episode_start", "timestamp": "2026-10-24T03:00:00Z", "task_id": "task_4", "seed": 42, "model": "test-model", "mode": "model"},
            {"type": "verification", "score": 0.0, "passed": False, "details": {"api_crash": "Connection timeout"}},
            {"type": "episode_end", "termination_reason": "crash", "final_score": 0.0, "total_steps": 5, "final_state_hash": "err"}
        ])
        runs = scan_run_logs(log_dirs=[self.logs_dir])
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0]["is_crash"])
        self.assertFalse(runs[0]["is_completed"])
        self.assertEqual(runs[0]["termination_reason"], "crash")

        summary = compute_summary({"task_4": {"difficulty": "medium"}}, {}, runs)
        self.assertEqual(summary["model_evaluations"]["crashed"], 1)
        self.assertEqual(summary["model_evaluations"]["completed"], 0)
        self.assertEqual(summary["model_evaluations"]["failed"], 0)
        self.assertIsNone(summary["model_evaluations"]["success_rate"])

    # 7. Multiple tasks and multiple seeds
    def test_07_multiple_tasks_and_seeds(self):
        for task_id in ["task_1", "task_2"]:
            for seed in [42, 99]:
                self._write_run(f"model_{task_id}_s{seed}.jsonl", [
                    {"type": "episode_start", "timestamp": "2026-10-24T04:00:00Z", "task_id": task_id, "seed": seed, "model": "test-model", "mode": "model"},
                    {"type": "episode_end", "termination_reason": "submitted", "final_score": 1.0, "total_steps": 10, "final_state_hash": "hash"}
                ])
        runs = scan_run_logs(log_dirs=[self.logs_dir])
        self.assertEqual(len(runs), 4)

        # Filtering
        t1_runs = scan_run_logs(log_dirs=[self.logs_dir], task_id_filter="task_1")
        self.assertEqual(len(t1_runs), 2)
        seeds = {r["seed"] for r in t1_runs}
        self.assertEqual(seeds, {42, 99})

    # 8. A changed task inventory
    def test_08_changed_task_inventory(self):
        custom_tasks = {
            "task_custom_1": {"difficulty": "easy"},
            "task_custom_2": {"difficulty": "hard"},
        }
        summary = compute_summary(custom_tasks, {}, [])
        self.assertEqual(summary["tasks"]["total"], 2)
        self.assertEqual(summary["tasks"]["difficulty_breakdown"]["easy"], 1)
        self.assertEqual(summary["tasks"]["difficulty_breakdown"]["hard"], 1)

    # 9. Missing optional metrics
    def test_09_missing_optional_metrics(self):
        self._write_run("sparse_run.jsonl", [
            {"type": "episode_start", "task_id": "task_1", "mode": "model"}
        ])
        runs = scan_run_logs(log_dirs=[self.logs_dir])
        self.assertEqual(len(runs), 1)
        self.assertIsNone(runs[0]["final_score"])
        self.assertFalse(runs[0]["passed"])

        summary = compute_summary({}, {}, runs)
        self.assertIsNone(summary["model_evaluations"]["average_score"])
        self.assertIsNone(summary["model_evaluations"]["success_rate"])

    # 10. Attempts to access paths outside approved result directories
    def test_10_path_traversal_prevention(self):
        self.assertIsNone(sanitize_filename("../../.env"))
        self.assertIsNone(sanitize_filename("..\\..\\passwords.txt"))
        self.assertIsNone(sanitize_filename("/etc/passwd"))
        self.assertIsNone(sanitize_filename("run_id/subpath"))
        self.assertIsNone(sanitize_filename("run_id\\subpath"))
        self.assertIsNone(sanitize_filename("run;rm -rf"))
        self.assertEqual(sanitize_filename("model_task_1_20261024"), "model_task_1_20261024")
        self.assertEqual(sanitize_filename("model_task_1.jsonl"), "model_task_1.jsonl")

        # Ensure get_run_trajectory returns None for path traversal attempts
        self.assertIsNone(get_run_trajectory("../../.env", search_dirs=[self.logs_dir]))
        self.assertIsNone(get_run_trajectory("..\\..\\Windows\\System32", search_dirs=[self.logs_dir]))


if __name__ == "__main__":
    unittest.main()
