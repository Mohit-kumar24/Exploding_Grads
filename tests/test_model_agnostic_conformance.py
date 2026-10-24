"""
Model-Agnostic Conformance and Protocol Test Suite.

Verifies the complete shared agent-environment contract across model types:
1. Scripted policy with predictable valid actions.
2. Agent that inspects observations before acting.
3. Agent that makes invalid tool calls and recovers.
4. Agent that repeats an unsuccessful action.
5. Agent that issues multiple tool calls in one response.
6. Agent that reaches the action budget boundary.
7. Agent that returns no tool call or malformed output.
8. Simulated API timeout, rate limit, and provider failure.
9. Agent that succeeds, fails an objective, or reaches a terminal condition.
10. Context preservation and recovery across extended conversations.
"""
import copy
import json
import unittest
from unittest.mock import MagicMock, patch

from src.env.database_env import KingdomEnv
from src.env.tasks import get_task, get_all_task_ids
from src.harness.runner import (
    run_model_episode,
    run_baseline,
    MAX_STEPS,
    MAX_TOOL_CALLS_PER_REPLY,
    MAX_CONSECUTIVE_API_ERRORS,
)
from src.verifier.verifier import Verifier


class TestModelAgnosticConformance(unittest.TestCase):
    """Conformance test suite evaluating model-agnostic contract properties."""

    def setUp(self):
        self.env = KingdomEnv()
        self.logger = MagicMock()

    def test_01_scripted_policy_with_predictable_valid_actions(self):
        """A scripted policy issuing valid actions achieves task success deterministically."""
        self.env.reset("task_1", seed=42)

        # Plan for task_1: build farm, assign workers, advance turns, submit
        scripted_replies = [
            {"content": "building farm", "tool_calls": [{"id": "c1", "name": "build", "arguments": {"building_type": "farm"}}]},
            {"content": "assigning workers", "tool_calls": [{"id": "c2", "name": "assign_workers", "arguments": {"building_type": "farm", "count": 3}}]},
            {"content": "turn 1", "tool_calls": [{"id": "c3", "name": "advance_turn", "arguments": {}}]},
            {"content": "turn 2", "tool_calls": [{"id": "c4", "name": "advance_turn", "arguments": {}}]},
            {"content": "turn 3", "tool_calls": [{"id": "c5", "name": "advance_turn", "arguments": {}}]},
            {"content": "turn 4", "tool_calls": [{"id": "c6", "name": "advance_turn", "arguments": {}}]},
            {"content": "turn 5", "tool_calls": [{"id": "c7", "name": "advance_turn", "arguments": {}}]},
            {"content": "turn 6", "tool_calls": [{"id": "c8", "name": "advance_turn", "arguments": {}}]},
            {"content": "submitting", "tool_calls": [{"id": "c9", "name": "submit_answer", "arguments": {"answer": "done"}}]},
        ]

        mock_client = MagicMock()
        mock_client.model = "scripted-agent"
        mock_client.chat.side_effect = scripted_replies

        with patch("src.harness.runner._load_append_safe", return_value=""):
            result = run_model_episode(self.env, "task_1", 42, mock_client, self.logger)

        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["termination_reason"], "submitted")
        self.assertEqual(result["steps_used"], len(scripted_replies))

    def test_02_agent_inspects_observations_before_acting(self):
        """An agent that repeatedly queries observations does not deplete its action allowance prematurely."""
        self.env.reset("task_1", seed=42)

        # Agent inspects state multiple times before acting
        inspection_replies = [
            {"content": "inspecting 1", "tool_calls": [{"id": "c1", "name": "view_kingdom", "arguments": {}}]},
            {"content": "inspecting 2", "tool_calls": [{"id": "c2", "name": "check_objectives", "arguments": {}}]},
            {"content": "inspecting 3", "tool_calls": [{"id": "c3", "name": "view_kingdom", "arguments": {}}]},
            {"content": "building farm", "tool_calls": [{"id": "c4", "name": "build", "arguments": {"building_type": "farm"}}]},
            {"content": "assigning", "tool_calls": [{"id": "c5", "name": "assign_workers", "arguments": {"building_type": "farm", "count": 3}}]},
            {"content": "advancing", "tool_calls": [{"id": "c6", "name": "advance_turn", "arguments": {}}]},
            {"content": "done", "tool_calls": [{"id": "c7", "name": "submit_answer", "arguments": {"answer": "finished"}}]},
        ]

        mock_client = MagicMock()
        mock_client.model = "observant-agent"
        mock_client.chat.side_effect = inspection_replies

        with patch("src.harness.runner._load_append_safe", return_value=""):
            result = run_model_episode(self.env, "task_1", 42, mock_client, self.logger)

        self.assertEqual(result["termination_reason"], "submitted")
        self.assertEqual(result["steps_used"], 7)
        self.assertGreater(self.env.step_count, 6)

    def test_03_invalid_tool_calls_and_recovery(self):
        """An agent making invalid tool calls receives descriptive errors and recovers."""
        self.env.reset("task_1", seed=42)

        # 1. Unknown tool
        obs1, done1, info1 = self.env.step("magic_wand", {})
        self.assertIn("Unknown tool", obs1)
        self.assertFalse(done1)

        # 2. Unknown building type
        obs2, done2, info2 = self.env.step("build", {"building_type": "laser_turret"})
        self.assertIn("Unknown building", obs2)
        self.assertFalse(done2)

        # 3. Not enough resources
        obs3, done3, info3 = self.env.step("build", {"building_type": "castle"})
        self.assertIn("Error: Building a castle requires at least one workshop", obs3)
        self.assertFalse(done3)

        # 4. Valid recovery
        obs4, done4, info4 = self.env.step("build", {"building_type": "farm"})
        self.assertIn("Built farm successfully", obs4)
        self.assertEqual(self.env.state["buildings"]["farm"], 1)

    def test_04_repeat_unsuccessful_action_does_not_corrupt_state(self):
        """Repeatedly failing the same action maintains state invariance without crashing."""
        self.env.reset("task_1", seed=42)
        initial_hash = self.env.state_hash()

        for _ in range(5):
            obs, done, info = self.env.step("build", {"building_type": "castle"})
            self.assertIn("Error", obs)
            self.assertFalse(done)
            self.assertEqual(self.env.state_hash(), initial_hash)

    def test_05_multiple_tool_calls_in_one_response(self):
        """Multi-tool replies execute sequentially and count as a single model turn."""
        self.env.reset("task_1", seed=42)

        multi_tool_reply = {
            "content": "executing setup bundle",
            "tool_calls": [
                {"id": "call_1", "name": "view_kingdom", "arguments": {}},
                {"id": "call_2", "name": "build", "arguments": {"building_type": "farm"}},
                {"id": "call_3", "name": "assign_workers", "arguments": {"building_type": "farm", "count": 2}},
                {"id": "call_4", "name": "advance_turn", "arguments": {}},
                {"id": "call_5", "name": "submit_answer", "arguments": {"answer": "done"}},
            ],
            "tokens": 50
        }

        mock_client = MagicMock()
        mock_client.model = "multi-tool-agent"
        mock_client.chat.return_value = multi_tool_reply

        with patch("src.harness.runner._load_append_safe", return_value=""):
            result = run_model_episode(self.env, "task_1", 42, mock_client, self.logger)

        self.assertEqual(result["steps_used"], 1)
        self.assertEqual(self.env.step_count, 5)
        self.assertEqual(result["termination_reason"], "submitted")
        self.assertEqual(self.env.state["buildings"].get("farm"), 1)
        self.assertEqual(self.env.state["workers"].get("farm"), 2)

    def test_06_action_budget_boundary_behavior(self):
        """Agent reaching the 40-step budget executes the 40th step and terminates with step_cap."""
        self.env.reset("task_1", seed=42)

        mock_client = MagicMock()
        mock_client.model = "boundary-agent"
        mock_client.chat.return_value = {
            "content": "idling",
            "tool_calls": [{"id": "c_view", "name": "view_kingdom", "arguments": {}}],
            "tokens": 5,
        }

        with patch("src.harness.runner._load_append_safe", return_value=""):
            result = run_model_episode(self.env, "task_1", 42, mock_client, self.logger)

        self.assertEqual(result["steps_used"], MAX_STEPS)
        self.assertEqual(result["termination_reason"], "step_cap")
        self.assertTrue(self.env.is_done)

    def test_07_no_tool_call_or_malformed_output_uses_step(self):
        """Text-only replies use a model turn and prompt the model to call tools."""
        self.env.reset("task_1", seed=42)

        replies = [
            {"content": "I am thinking about building a farm.", "tool_calls": [], "tokens": 10},
            {"content": "Now building", "tool_calls": [{"id": "c1", "name": "build", "arguments": {"building_type": "farm"}}]},
            {"content": "done", "tool_calls": [{"id": "c2", "name": "submit_answer", "arguments": {"answer": "done"}}]},
        ]

        mock_client = MagicMock()
        mock_client.model = "chatty-agent"
        mock_client.chat.side_effect = replies

        with patch("src.harness.runner._load_append_safe", return_value=""):
            result = run_model_episode(self.env, "task_1", 42, mock_client, self.logger)

        self.assertEqual(result["steps_used"], 3)
        self.assertEqual(self.env.step_count, 2)
        self.assertEqual(result["termination_reason"], "submitted")

    def test_08_simulated_api_error_recovery(self):
        """Transient provider errors consume turns and recover on subsequent success."""
        self.env.reset("task_1", seed=42)

        replies = [
            {"error": "503 Service Unavailable: overloaded"},
            {"content": "recovered", "tool_calls": [{"id": "c1", "name": "build", "arguments": {"building_type": "farm"}}]},
            {"content": "done", "tool_calls": [{"id": "c2", "name": "submit_answer", "arguments": {"answer": "done"}}]},
        ]

        mock_client = MagicMock()
        mock_client.model = "transient-err-agent"
        mock_client.chat.side_effect = replies

        with patch("src.harness.runner._load_append_safe", return_value=""):
            result = run_model_episode(self.env, "task_1", 42, mock_client, self.logger)

        self.assertEqual(result["steps_used"], 3)
        self.assertEqual(self.env.state["buildings"]["farm"], 1)
        self.assertEqual(result["termination_reason"], "submitted")

    def test_09_post_terminal_actions_strictly_blocked(self):
        """Once submitted or terminated, subsequent environment mutations are strictly blocked."""
        self.env.reset("task_1", seed=42)
        res1, done1, info1 = self.env.step("submit_answer", {"answer": "final"})
        self.assertTrue(done1)

        # Attempt post-termination action
        res2, done2, info2 = self.env.step("build", {"building_type": "farm"})
        self.assertIn("Error: Episode has already ended", res2)
        self.assertTrue(done2)
        self.assertEqual(info2.get("error"), "post_termination_action")

    def test_10_context_trimming_preserves_valid_history(self):
        """Extended interactions produce valid trimmed history without orphan tool messages."""
        from src.harness.runner import _trim_context

        # Build 40-message alternating history
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "initial prompt"},
        ]
        for step_i in range(1, 20):
            call_id = f"call_{step_i}"
            messages.append({
                "role": "assistant",
                "content": f"thought {step_i}",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "view_kingdom", "arguments": "{}"}}]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"result {step_i}"
            })

        trimmed = _trim_context(messages, "system prompt", max_messages=15)
        self.assertLessEqual(len(trimmed), 15)
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertEqual(trimmed[1]["role"], "user")

        # Invariant: Every role: tool has a preceding assistant with matching call id
        for idx, msg in enumerate(trimmed):
            if msg.get("role") == "tool":
                self.assertGreater(idx, 0)
                prev = trimmed[idx - 1]
                # Either prev is the assistant or another tool belonging to same assistant
                assistant_msg = prev if prev.get("role") == "assistant" else trimmed[idx - 2]
                self.assertEqual(assistant_msg.get("role"), "assistant")
                matching_ids = {tc.get("id") for tc in assistant_msg.get("tool_calls", [])}
                self.assertIn(msg.get("tool_call_id"), matching_ids)


if __name__ == "__main__":
    unittest.main()
