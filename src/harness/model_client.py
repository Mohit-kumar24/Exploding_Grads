"""
Model Client -- OpenAI-compatible API interface

Handles communication with any model through the standard chat completions API.
Configurable via MODEL, API_KEY, and BASE_URL environment variables.
Handles errors, timeouts, malformed responses, and multiple tool calls.
"""
import json
import time
import os
from typing import Any, Dict, List, Optional, Tuple


class ModelClient:
    """OpenAI-compatible model client."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
        self._total_tokens = 0
        self._total_calls = 0

    def _get_client(self):
        """Lazy-initialize the generic OpenAI-compatible client."""
        if self._client is None:
            try:
                import openai
                clean_base_url = self.base_url.strip() if self.base_url and self.base_url.strip() else None
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=clean_base_url,
                    timeout=90.0,
                )
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    def _redact_secrets(self, text: str) -> str:
        """Ensure active API key and credentials never leak in error messages."""
        if not text:
            return ""
        if self.api_key and len(self.api_key) >= 6:
            text = text.replace(self.api_key, "[REDACTED_KEY]")
        import re
        text = re.sub(r'(sk-[a-zA-Z0-9_\-]{15,})', '[REDACTED_KEY]', text)
        text = re.sub(r'(bearer\s+)\S{15,}', r'\1[REDACTED]', text, flags=re.IGNORECASE)
        return text

    def chat(self, messages: List[Dict], tools: List[Dict],
             timeout: float = 90.0) -> Dict[str, Any]:
        """
        Send a chat completion request to any OpenAI-compatible provider.

        Returns dict with:
        - content: text content of reply (may be None)
        - tool_calls: list of tool call dicts [{id, name, arguments}, ...]
        - raw: raw response string
        - tokens: token usage
        - latency_ms: response time
        - error: error string if any
        """
        client = self._get_client()
        start = time.time()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                timeout=timeout,
            )

            latency_ms = (time.time() - start) * 1000

            # Support both Pydantic models (openai SDK) and dict structures (proxies/gateways)
            choices = getattr(response, "choices", None) or (response.get("choices") if isinstance(response, dict) else None)
            choice = choices[0] if choices else None

            if choice is None:
                err_obj = getattr(response, "error", None) or (response.get("error") if isinstance(response, dict) else None)
                if err_obj:
                    err_msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                    return {
                        "content": None,
                        "tool_calls": [],
                        "raw": str(response)[:2000],
                        "tokens": 0,
                        "latency_ms": latency_ms,
                        "error": "Provider error: %s" % self._redact_secrets(str(err_msg)),
                    }
                return {
                    "content": None,
                    "tool_calls": [],
                    "raw": str(response)[:2000],
                    "tokens": 0,
                    "latency_ms": latency_ms,
                    "error": "Empty response from model (no choices returned)",
                }

            message = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
            if message is None:
                return {
                    "content": None,
                    "tool_calls": [],
                    "raw": str(response)[:2000],
                    "tokens": 0,
                    "latency_ms": latency_ms,
                    "error": "Malformed response: missing message in choice",
                }

            content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")

            # Parse tool calls across provider variants
            tool_calls = self._parse_tool_calls(message, content)

            # Token usage
            tokens = 0
            usage = getattr(response, "usage", None) or (response.get("usage") if isinstance(response, dict) else None)
            if usage:
                tokens = getattr(usage, "total_tokens", None) or (usage.get("total_tokens") if isinstance(usage, dict) else 0) or 0

            self._total_tokens += tokens
            self._total_calls += 1

            return {
                "content": content,
                "tool_calls": tool_calls,
                "raw": str(response)[:2000],
                "tokens": tokens,
                "latency_ms": latency_ms,
                "error": None,
            }

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            safe_error = self._redact_secrets(str(e))
            return {
                "content": None,
                "tool_calls": [],
                "raw": "",
                "tokens": 0,
                "latency_ms": latency_ms,
                "error": "API error: %s" % safe_error,
            }

    def _parse_tool_calls(self, message: Any, content: Any) -> List[Dict[str, Any]]:
        """
        Robustly extract tool calls from provider message.
        Handles:
        1. Standard OpenAI tool_calls list (JSON string or pre-parsed dict).
        2. Flattened tool calls without nested function object.
        3. Legacy single function_call.
        4. Embedded JSON tool calls in text content when native calling is omitted.
        """
        tool_calls = []
        raw_tool_calls = getattr(message, "tool_calls", None) or (message.get("tool_calls") if isinstance(message, dict) else None)

        if raw_tool_calls:
            for idx, tc in enumerate(raw_tool_calls):
                tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None) or ("call_%d" % idx)
                tc_func = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)

                if tc_func is not None:
                    tc_name = getattr(tc_func, "name", None) or (tc_func.get("name") if isinstance(tc_func, dict) else "")
                    raw_args = getattr(tc_func, "arguments", None) or (tc_func.get("arguments") if isinstance(tc_func, dict) else None)
                else:
                    tc_name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "")
                    raw_args = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else None)

                # Parse arguments
                if isinstance(raw_args, dict):
                    parsed_args = raw_args
                elif isinstance(raw_args, str):
                    if not raw_args.strip():
                        parsed_args = {}
                    else:
                        try:
                            parsed_args = json.loads(raw_args)
                            if not isinstance(parsed_args, dict):
                                parsed_args = {"_parse_error": "Arguments must be a JSON object, got %s" % type(parsed_args).__name__}
                        except Exception:
                            parsed_args = {"_parse_error": raw_args}
                elif raw_args is None:
                    parsed_args = {}
                else:
                    parsed_args = {"_parse_error": str(raw_args)}

                tool_calls.append({
                    "id": str(tc_id),
                    "name": str(tc_name).strip(),
                    "arguments": parsed_args,
                })

        # Fallback: check for legacy function_call
        if not tool_calls:
            raw_fc = getattr(message, "function_call", None) or (message.get("function_call") if isinstance(message, dict) else None)
            if raw_fc:
                fc_name = getattr(raw_fc, "name", None) or (raw_fc.get("name") if isinstance(raw_fc, dict) else "")
                raw_args = getattr(raw_fc, "arguments", None) or (raw_fc.get("arguments") if isinstance(raw_fc, dict) else None)
                if isinstance(raw_args, dict):
                    parsed_args = raw_args
                elif isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                    except Exception:
                        parsed_args = {"_parse_error": raw_args}
                else:
                    parsed_args = {}
                if fc_name:
                    tool_calls.append({
                        "id": "call_function_0",
                        "name": str(fc_name).strip(),
                        "arguments": parsed_args,
                    })

        # Fallback: extract valid JSON tool call embedded in content if native calling produced nothing
        if not tool_calls and content and isinstance(content, str):
            extracted = self._extract_tool_call_from_content(content)
            if extracted:
                tool_calls.extend(extracted)

        return tool_calls

    def _extract_tool_call_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool call formatted as JSON in message text content."""
        content_stripped = content.strip()
        if not (content_stripped.startswith("{") and content_stripped.endswith("}")):
            return []
        try:
            data = json.loads(content_stripped)
            if isinstance(data, dict):
                tool_name = data.get("name") or data.get("tool")
                tool_args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
                if tool_name and isinstance(tool_name, str):
                    from src.env.tools import VALID_TOOL_NAMES
                    if tool_name in VALID_TOOL_NAMES:
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except Exception:
                                tool_args = {"_parse_error": tool_args}
                        if not isinstance(tool_args, dict):
                            tool_args = {}
                        return [{
                            "id": "content_call_0",
                            "name": tool_name,
                            "arguments": tool_args,
                        }]
        except Exception:
            pass
        return []

    @property
    def total_tokens(self):
        return self._total_tokens

    @property
    def total_calls(self):
        return self._total_calls


class MockModelClient:
    """
    Mock model client for offline testing.
    Returns predefined responses or empty responses.
    """

    def __init__(self, responses: Optional[List[Dict]] = None):
        self._responses = responses or []
        self._call_index = 0
        self._total_tokens = 0
        self._total_calls = 0
        self.model = "mock-model"

    def chat(self, messages: List[Dict], tools: List[Dict],
             timeout: float = 90.0) -> Dict[str, Any]:
        self._total_calls += 1

        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp

        # Default: return empty response
        return {
            "content": "I don't know what to do.",
            "tool_calls": [],
            "raw": "",
            "tokens": 10,
            "latency_ms": 50.0,
            "error": None,
        }

    @property
    def total_tokens(self):
        return self._total_tokens

    @property
    def total_calls(self):
        return self._total_calls
