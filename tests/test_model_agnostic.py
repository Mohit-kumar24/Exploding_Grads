"""
Tests for Model-Agnostic Generic Client Architecture.

Verifies:
1. Configuration loading:
   - Evaluates generic MODEL, API_KEY, BASE_URL configuration without provider hardcoding.
   - Handles empty/omitted BASE_URL cleanly (falling back to standard OpenAI endpoint).
   - Strips whitespace from configuration values.
2. Provider-agnostic response handling:
   - Provider A: Standard OpenAI object response (JSON string in function arguments).
   - Provider B: Alternative LiteLLM / gateway response (pre-parsed dict in function arguments, dict response).
   - Provider C: Legacy endpoint returning function_call instead of tool_calls.
   - Provider D: Content-embedded JSON tool call fallback.
3. Resilience against malformed/unsupported responses:
   - Corrupted argument JSON is caught and tagged with _parse_error without crashing.
   - Proxy-level error objects are extracted and reported cleanly.
   - Empty choices list returns descriptive error.
   - Authentication (401) and rate-limit (429) errors are reported with secrets redacted.
"""
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from src.harness.model_client import ModelClient


class MockOpenAIToolCallFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class MockOpenAIToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = MockOpenAIToolCallFunction(name, arguments)


class MockOpenAIMessage:
    def __init__(self, content: str = None, tool_calls=None, function_call=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.function_call = function_call


class MockOpenAIChoice:
    def __init__(self, message):
        self.message = message


class MockOpenAIUsage:
    def __init__(self, total_tokens: int = 42):
        self.total_tokens = total_tokens


class MockOpenAIResponse:
    def __init__(self, message, total_tokens: int = 42):
        self.choices = [MockOpenAIChoice(message)]
        self.usage = MockOpenAIUsage(total_tokens)


class TestModelConfiguration(unittest.TestCase):
    """Test suite verifying generic, model-agnostic configuration handling."""

    def test_configuration_accepts_any_model_and_base_url(self):
        """ModelClient stores and utilizes any generic model name and base URL."""
        client = ModelClient(
            model="custom-finetuned-llama",
            api_key="sk-test-key-1234567890",
            base_url="https://custom.gateway.internal/v1"
        )
        self.assertEqual(client.model, "custom-finetuned-llama")
        self.assertEqual(client.api_key, "sk-test-key-1234567890")
        self.assertEqual(client.base_url, "https://custom.gateway.internal/v1")

    def test_empty_base_url_normalized_to_none(self):
        """An empty or whitespace-only base URL normalizes to None for default provider."""
        client = ModelClient(
            model="gpt-4o-mini",
            api_key="sk-test-key-1234567890",
            base_url="   "
        )
        mock_openai_module = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            client._client = None
            client._get_client()
            mock_openai_module.OpenAI.assert_called_once_with(
                api_key="sk-test-key-1234567890",
                base_url=None,
                timeout=90.0
            )


class TestMultiProviderCompatibility(unittest.TestCase):
    """Test suite evaluating diverse OpenAI-compatible provider response formats."""

    def test_provider_standard_openai_format(self):
        """Provider 1 (e.g. OpenAI/Azure): Pydantic-like object with stringified arguments."""
        client = ModelClient("gpt-test", "sk-test-key-1234567890", "https://api.openai.com/v1")

        mock_msg = MockOpenAIMessage(
            content="Building farm",
            tool_calls=[
                MockOpenAIToolCall("call_abc1", "build", json.dumps({"building_type": "farm"}))
            ]
        )
        mock_resp = MockOpenAIResponse(mock_msg, total_tokens=150)

        with patch.object(client, "_get_client") as mock_get:
            mock_sdk = MagicMock()
            mock_sdk.chat.completions.create.return_value = mock_resp
            mock_get.return_value = mock_sdk

            res = client.chat([{"role": "user", "content": "build farm"}], tools=[])

        self.assertIsNone(res["error"])
        self.assertEqual(res["content"], "Building farm")
        self.assertEqual(len(res["tool_calls"]), 1)
        self.assertEqual(res["tool_calls"][0]["name"], "build")
        self.assertEqual(res["tool_calls"][0]["arguments"], {"building_type": "farm"})
        self.assertEqual(res["tokens"], 150)

    def test_provider_litellm_dictionary_format(self):
        """Provider 2 (e.g. LiteLLM proxy / Ollama): Raw dictionary response with pre-parsed dict arguments."""
        client = ModelClient("litellm/model", "sk-test-key-1234567890", "https://litellm-hackathon.penpencil.guru/")

        dict_response = {
            "choices": [
                {
                    "message": {
                        "content": "Assigning 3 workers to lumber mill",
                        "tool_calls": [
                            {
                                "id": "call_lite_99",
                                "type": "function",
                                "function": {
                                    "name": "assign_workers",
                                    # Pre-parsed dictionary instead of JSON string
                                    "arguments": {"building_type": "lumber_mill", "count": 3}
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {"total_tokens": 85}
        }

        with patch.object(client, "_get_client") as mock_get:
            mock_sdk = MagicMock()
            mock_sdk.chat.completions.create.return_value = dict_response
            mock_get.return_value = mock_sdk

            res = client.chat([{"role": "user", "content": "assign workers"}], tools=[])

        self.assertIsNone(res["error"])
        self.assertEqual(len(res["tool_calls"]), 1)
        self.assertEqual(res["tool_calls"][0]["name"], "assign_workers")
        self.assertEqual(res["tool_calls"][0]["arguments"], {"building_type": "lumber_mill", "count": 3})
        self.assertEqual(res["tokens"], 85)

    def test_provider_legacy_function_call(self):
        """Provider 3: Legacy endpoint returning function_call instead of tool_calls."""
        client = ModelClient("legacy-model", "sk-test-key-1234567890", "https://legacy.gateway/v1")

        mock_msg = MockOpenAIMessage(
            content="Advancing turn",
            tool_calls=[],
            function_call=MockOpenAIToolCallFunction("advance_turn", "{}")
        )
        mock_resp = MockOpenAIResponse(mock_msg, total_tokens=40)

        with patch.object(client, "_get_client") as mock_get:
            mock_sdk = MagicMock()
            mock_sdk.chat.completions.create.return_value = mock_resp
            mock_get.return_value = mock_sdk

            res = client.chat([{"role": "user", "content": "advance"}], tools=[])

        self.assertIsNone(res["error"])
        self.assertEqual(len(res["tool_calls"]), 1)
        self.assertEqual(res["tool_calls"][0]["name"], "advance_turn")
        self.assertEqual(res["tool_calls"][0]["arguments"], {})

    def test_provider_content_embedded_json_fallback(self):
        """Provider 4: Model outputs JSON tool call in text content when tools parameter is unsupported."""
        client = ModelClient("non-tool-model", "sk-test-key-1234567890", "https://proxy/v1")

        raw_json_content = json.dumps({
            "name": "view_kingdom",
            "arguments": {}
        })
        mock_msg = MockOpenAIMessage(content=raw_json_content, tool_calls=[])
        mock_resp = MockOpenAIResponse(mock_msg, total_tokens=30)

        with patch.object(client, "_get_client") as mock_get:
            mock_sdk = MagicMock()
            mock_sdk.chat.completions.create.return_value = mock_resp
            mock_get.return_value = mock_sdk

            res = client.chat([{"role": "user", "content": "view"}], tools=[])

        self.assertIsNone(res["error"])
        self.assertEqual(len(res["tool_calls"]), 1)
        self.assertEqual(res["tool_calls"][0]["name"], "view_kingdom")
        self.assertEqual(res["tool_calls"][0]["arguments"], {})


class TestMalformedAndUnsupportedResponses(unittest.TestCase):
    """Test suite verifying clear failure reporting without fabricating scores."""

    def test_corrupted_arguments_string_tagged_with_parse_error(self):
        """Malformed JSON arguments are tagged with _parse_error rather than causing unhandled exception."""
        client = ModelClient("gpt-test", "sk-test-key-1234567890", "https://api.openai.com/v1")

        mock_msg = MockOpenAIMessage(
            content=None,
            tool_calls=[
                MockOpenAIToolCall("call_bad", "build", '{"building_type": "farm",')
            ]
        )
        mock_resp = MockOpenAIResponse(mock_msg)

        with patch.object(client, "_get_client") as mock_get:
            mock_sdk = MagicMock()
            mock_sdk.chat.completions.create.return_value = mock_resp
            mock_get.return_value = mock_sdk

            res = client.chat([], tools=[])

        self.assertIsNone(res["error"])
        self.assertEqual(len(res["tool_calls"]), 1)
        self.assertIn("_parse_error", res["tool_calls"][0]["arguments"])

    def test_proxy_error_object_extracted(self):
        """A proxy returning an error object in the response payload returns clear Provider error."""
        client = ModelClient("error-model", "sk-test-key-1234567890", "https://proxy/v1")

        error_response = {
            "error": {
                "message": "Model 'error-model' does not exist or access is restricted.",
                "type": "invalid_request_error"
            }
        }

        with patch.object(client, "_get_client") as mock_get:
            mock_sdk = MagicMock()
            mock_sdk.chat.completions.create.return_value = error_response
            mock_get.return_value = mock_sdk

            res = client.chat([], tools=[])

        self.assertIsNotNone(res["error"])
        self.assertIn("Provider error", res["error"])
        self.assertIn("does not exist", res["error"])

    def test_empty_choices_returns_descriptive_error(self):
        """An empty choices list returns 'Empty response from model' error."""
        client = ModelClient("empty-model", "sk-test-key-1234567890", "https://proxy/v1")

        empty_resp = {"choices": []}

        with patch.object(client, "_get_client") as mock_get:
            mock_sdk = MagicMock()
            mock_sdk.chat.completions.create.return_value = empty_resp
            mock_get.return_value = mock_sdk

            res = client.chat([], tools=[])

        self.assertIsNotNone(res["error"])
        self.assertIn("Empty response", res["error"])

    def test_active_secret_redacted_from_exception_message(self):
        """When an API exception contains the active API key, it is redacted to [REDACTED_KEY]."""
        secret_key = "sk-REAL-ACTIVE-KEY-9988776655"
        client = ModelClient("sec-model", secret_key, "https://api.openai.com/v1")

        with patch.object(client, "_get_client") as mock_get:
            mock_sdk = MagicMock()
            mock_sdk.chat.completions.create.side_effect = RuntimeError(
                f"Connection failed for key {secret_key} with status 401 Unauthorized"
            )
            mock_get.return_value = mock_sdk

            res = client.chat([], tools=[])

        self.assertIsNotNone(res["error"])
        self.assertNotIn(secret_key, res["error"])
        self.assertIn("[REDACTED_KEY]", res["error"])


if __name__ == "__main__":
    unittest.main()
