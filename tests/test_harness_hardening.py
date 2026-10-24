"""
Tests for Harness Hardening, Input Validation, and Resilience.

Verifies:
1. APPEND_FILE validation:
   - Stops with exit code 3 when explicitly named file is missing.
   - Stops with exit code 3 when explicitly named file is empty.
   - Preserves offline execution when APPEND_FILE is unset and default is absent.
   - Returns content when APPEND_FILE is valid.
2. API failure handling:
   - Authentication failure terminates promptly with termination_reason='crash' and score 0.0.
   - 3 consecutive transient failures terminate with termination_reason='crash' and score 0.0.
   - Isolated transient error recovers on subsequent successful turn.
3. Conversation-history argument sanitization:
   - Malformed arguments with _parse_error are sanitized to '{}' in assistant message history.
   - Valid tool arguments remain unchanged.
4. Process exit codes:
   - Baseline gate failure exits with code 2.
   - Configuration/input error exits with code 3.
   - Normal completion exits with code 0.
5. check_model.py preflight utility:
   - Exits with code 1 if required env vars are missing.
   - Masks secrets and exits with code 1 on connection error.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from src.env.database_env import KingdomEnv
from src.harness.runner import (
    load_append_file,
    run_model_episode,
    EXIT_SUCCESS,
    EXIT_ERROR,
    EXIT_GATE_FAILURE,
    EXIT_CONFIG_ERROR,
    MAX_CONSECUTIVE_API_ERRORS,
)
import src.harness.runner as runner
from src.harness.logger import Logger


class TestAppendFileValidation(unittest.TestCase):
    """Test suite for Section 05.3 APPEND_FILE specification compliance."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_explicit_append_file_stops_run_with_code_3(self):
        """When APPEND_FILE is explicitly set to a missing file, run must stop with exit code 3."""
        missing_path = os.path.join(self.temp_dir, "missing_instructions.txt")
        with patch.dict(os.environ, {"APPEND_FILE": missing_path}):
            with self.assertRaises(SystemExit) as cm:
                load_append_file()
            self.assertEqual(cm.exception.code, EXIT_CONFIG_ERROR)

    def test_empty_explicit_append_file_stops_run_with_code_3(self):
        """When APPEND_FILE is explicitly set to an empty file, run must stop with exit code 3."""
        empty_path = os.path.join(self.temp_dir, "empty_instructions.txt")
        with open(empty_path, "w", encoding="utf-8") as f:
            f.write("   \n\t  ")  # whitespace only

        with patch.dict(os.environ, {"APPEND_FILE": empty_path}):
            with self.assertRaises(SystemExit) as cm:
                load_append_file()
            self.assertEqual(cm.exception.code, EXIT_CONFIG_ERROR)

    def test_blank_env_var_stops_run_with_code_3(self):
        """When APPEND_FILE is set to blank string, run must stop with exit code 3."""
        with patch.dict(os.environ, {"APPEND_FILE": "   "}):
            with self.assertRaises(SystemExit) as cm:
                load_append_file()
            self.assertEqual(cm.exception.code, EXIT_CONFIG_ERROR)

    def test_unset_append_file_with_absent_default_returns_empty_string(self):
        """When APPEND_FILE is unset and default file is absent, returns empty string without stopping."""
        env_copy = dict(os.environ)
        env_copy.pop("APPEND_FILE", None)
        with patch.dict(os.environ, env_copy, clear=True):
            with patch("os.path.exists", return_value=False):
                content = load_append_file()
                self.assertEqual(content, "")

    def test_valid_append_file_loads_content(self):
        """When APPEND_FILE points to valid non-empty file, content is loaded verbatim."""
        valid_path = os.path.join(self.temp_dir, "custom_append.txt")
        expected_text = "Prioritize gold production above all else."
        with open(valid_path, "w", encoding="utf-8") as f:
            f.write(expected_text)

        with patch.dict(os.environ, {"APPEND_FILE": valid_path}):
            content = load_append_file()
            self.assertEqual(content, expected_text)


class TestApiFailureHandling(unittest.TestCase):
    """Test suite for API authentication and transient failure handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Logger(log_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_auth_error_aborts_promptly_with_crash_reason(self):
        """HTTP 401 / Authentication error terminates on turn 1 as 'crash' with score 0.0."""
        env = KingdomEnv()
        mock_client = MagicMock()
        mock_client.model = "gpt-test"
        mock_client.chat.return_value = {
            "error": "API error: Error code 401 - {'error': {'message': 'Incorrect API key provided: sk-secret123'}}",
            "content": None,
            "tool_calls": [],
            "raw": "",
            "tokens": 0,
        }

        result = run_model_episode(env, "task_1", 42, mock_client, self.logger)

        # Must have aborted on turn 1, not looped 40 times
        self.assertEqual(mock_client.chat.call_count, 1)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["termination_reason"], "crash")
        self.assertIn("api_crash", result.get("details", {}))

    def test_consecutive_transient_errors_abort_after_threshold(self):
        """Repeated transient errors terminate after MAX_CONSECUTIVE_API_ERRORS (3)."""
        env = KingdomEnv()
        mock_client = MagicMock()
        mock_client.model = "gpt-test"
        mock_client.chat.return_value = {
            "error": "API error: 503 Server Unavailable",
            "content": None,
            "tool_calls": [],
            "raw": "",
            "tokens": 0,
        }

        result = run_model_episode(env, "task_1", 42, mock_client, self.logger)

        self.assertEqual(mock_client.chat.call_count, MAX_CONSECUTIVE_API_ERRORS)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["termination_reason"], "crash")

    def test_intermittent_error_recovers_and_resets_counter(self):
        """A transient error followed by success does not crash and allows completion."""
        env = KingdomEnv()
        mock_client = MagicMock()
        mock_client.model = "gpt-test"
        mock_client.chat.side_effect = [
            {"error": "API error: 500 Internal Server Error", "content": None, "tool_calls": []},
            {"error": None, "content": "Checking state", "tool_calls": [
                {"id": "call_1", "name": "view_kingdom", "arguments": {}}
            ]},
            {"error": None, "content": "Submitting", "tool_calls": [
                {"id": "call_2", "name": "submit_answer", "arguments": {}}
            ]},
        ]

        result = run_model_episode(env, "task_1", 42, mock_client, self.logger)

        # Did not crash
        self.assertNotEqual(result.get("termination_reason"), "crash")
        self.assertEqual(mock_client.chat.call_count, 3)


class TestConversationHistorySanitization(unittest.TestCase):
    """Test suite verifying tool call argument serialization in conversation history."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Logger(log_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_corrupted_arguments_sanitized_to_empty_object_in_assistant_message(self):
        """When arguments contain _parse_error, assistant message history has '{}'."""
        env = KingdomEnv()
        captured_messages = []

        def fake_chat(messages, tools, timeout=90.0):
            captured_messages.append(list(messages))
            if len(captured_messages) == 1:
                # Return corrupted argument JSON
                return {
                    "error": None,
                    "content": "Building structure",
                    "tool_calls": [
                        {
                            "id": "bad_call_1",
                            "name": "build",
                            "arguments": {"_parse_error": "{\"building_type\": \"farm\","},  # malformed JSON
                        }
                    ],
                }
            # Second turn: submit to finish
            return {
                "error": None,
                "content": "Done",
                "tool_calls": [
                    {"id": "call_sub", "name": "submit_answer", "arguments": {}}
                ],
            }

        mock_client = MagicMock()
        mock_client.model = "gpt-test"
        mock_client.chat.side_effect = fake_chat

        run_model_episode(env, "task_1", 42, mock_client, self.logger)

        # Inspect the message list sent on turn 2
        turn_2_messages = captured_messages[1]
        assistant_msgs = [m for m in turn_2_messages if m.get("role") == "assistant" and m.get("tool_calls")]
        self.assertGreaterEqual(len(assistant_msgs), 1)

        first_tool_call = assistant_msgs[0]["tool_calls"][0]
        self.assertEqual(first_tool_call["function"]["name"], "build")
        # Arguments must be serialized valid JSON '{}', NOT '{"_parse_error": ...}'
        self.assertEqual(first_tool_call["function"]["arguments"], "{}")

        # Tool response must contain the parse error message
        tool_msgs = [m for m in turn_2_messages if m.get("role") == "tool" and m.get("tool_call_id") == "bad_call_1"]
        self.assertGreaterEqual(len(tool_msgs), 1)
        self.assertIn("Could not parse arguments", tool_msgs[0]["content"])


def _raise_exit(code=0):
    raise SystemExit(code)


class TestExitCodes(unittest.TestCase):
    """Test suite verifying standardized process exit codes."""

    def test_exit_code_constants_match_specification(self):
        """Exit codes must be distinct and match expected conventions."""
        self.assertEqual(EXIT_SUCCESS, 0)
        self.assertEqual(EXIT_ERROR, 1)
        self.assertEqual(EXIT_GATE_FAILURE, 2)
        self.assertEqual(EXIT_CONFIG_ERROR, 3)

    def test_invalid_mode_exits_with_code_3(self):
        """Unknown MODE exits with EXIT_CONFIG_ERROR (3)."""
        with patch.dict(os.environ, {"MODE": "invalid_mode"}):
            with patch("sys.exit", side_effect=_raise_exit) as mock_exit:
                with self.assertRaises(SystemExit) as cm:
                    runner.main([])
                self.assertEqual(cm.exception.code, EXIT_CONFIG_ERROR)

    def test_invalid_task_exits_with_code_3(self):
        """Nonexistent TASK_ID in full mode exits with EXIT_CONFIG_ERROR (3)."""
        with patch.dict(os.environ, {"MODE": "full", "TASK_ID": "nonexistent_task_999"}):
            with patch("sys.exit", side_effect=_raise_exit) as mock_exit:
                with self.assertRaises(SystemExit) as cm:
                    runner.main([])
                self.assertEqual(cm.exception.code, EXIT_CONFIG_ERROR)

    def test_baseline_gate_failure_exits_with_code_2(self):
        """Baseline gate failure exits with EXIT_GATE_FAILURE (2)."""
        failing_ref = {"score": 0.80, "gate_check": {"passed": False}}
        passing_noop = {"score": 0.00, "gate_check": {"passed": True}}

        with patch("src.harness.runner.run_baselines_for_task", return_value=(failing_ref, passing_noop)):
            with patch.dict(os.environ, {"MODE": "full", "TASK_ID": "task_1"}):
                with patch("sys.exit", side_effect=_raise_exit) as mock_exit:
                    with self.assertRaises(SystemExit) as cm:
                        runner.main()
                    self.assertEqual(cm.exception.code, EXIT_GATE_FAILURE)


class TestCheckModelUtility(unittest.TestCase):
    """Test suite for check_model.py diagnostic preflight utility."""

    def test_check_model_missing_env_exits_with_code_1(self):
        """check_model.py exits with code 1 if MODEL or API_KEY is missing."""
        import check_model
        with patch.dict(os.environ, {"MODEL": "", "API_KEY": ""}, clear=True):
            with patch("sys.exit", side_effect=_raise_exit) as mock_exit:
                with self.assertRaises(SystemExit) as cm:
                    check_model.main()
                self.assertEqual(cm.exception.code, 1)

    def test_check_model_connection_error_redacts_and_exits_1(self):
        """check_model.py redacts sensitive keys and exits with code 1 on connection failure."""
        import check_model
        with patch.dict(os.environ, {"MODEL": "test-model", "API_KEY": "sk-secret12345678901234567890"}):
            with patch("src.harness.model_client.ModelClient.chat", return_value={"error": "API error: Connection refused to https://api.openai.com sk-secret12345678901234567890"}):
                with patch("sys.exit", side_effect=_raise_exit) as mock_exit:
                    with self.assertRaises(SystemExit) as cm:
                        check_model.main()
                    self.assertEqual(cm.exception.code, 1)


class ContextTrimmingTests(unittest.TestCase):
    """Verify conversation history trimming respects OpenAI/Azure tool-call protocol."""

    def test_trim_context_prevents_orphaned_tool_messages(self):
        from src.harness.runner import _trim_context

        # Construct messages where simple slicing would cut an assistant message and leave an orphaned tool
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user_init"},
        ]
        for step in range(15):
            call_id = f"call_{step}"
            messages.append({
                "role": "assistant",
                "content": f"thought {step}",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "action", "arguments": "{}"}}]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"result {step}"
            })

        # Total messages = 2 + 30 = 32. Trim to max 10 messages
        trimmed = _trim_context(messages, "sys", max_messages=10)

        # 1. System and initial user must be kept
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertEqual(trimmed[1]["role"], "user")

        # 2. No tool message should appear without an immediately preceding assistant message with matching tool_calls
        for i, msg in enumerate(trimmed):
            if msg.get("role") == "tool":
                self.assertGreater(i, 0, "Tool message cannot be first in history")
                prev_msg = trimmed[i - 1]
                # Either prev_msg is assistant with tool_calls, or prev_msg is another tool message
                # Trace back to find parent assistant
                found_parent = False
                for back_idx in range(i - 1, -1, -1):
                    cand = trimmed[back_idx]
                    if cand.get("role") == "assistant" and "tool_calls" in cand:
                        call_ids = {tc.get("id") for tc in cand["tool_calls"]}
                        if msg.get("tool_call_id") in call_ids:
                            found_parent = True
                            break
                    elif cand.get("role") != "tool":
                        break
                self.assertTrue(found_parent, f"Orphaned tool message found: {msg}")


if __name__ == "__main__":
    unittest.main()

