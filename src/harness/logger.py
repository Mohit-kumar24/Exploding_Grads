"""
Structured JSONL Logger

Writes structured log entries to ./logs/ directory.
Each episode gets its own log file.
Secrets are redacted before logging.
"""
import json
import os
import time
import re
from typing import Any, Dict, Optional

try:
    from src.harness.terminal import colorize
except ImportError:
    try:
        from .terminal import colorize
    except ImportError:
        def colorize(text: str, color: str = None, bold: bool = False) -> str:
            return text


class Logger:
    """Structured JSONL logger for episodes."""

    def __init__(self, log_dir: str = "./logs"):
        self._log_dir = log_dir
        self._current_file = None
        self._current_path = None
        self._run_id = None
        os.makedirs(log_dir, exist_ok=True)

    def start_episode(self, run_id: str, task_id: str, seed: int,
                      model: str = "", mode: str = ""):
        """Start logging a new episode."""
        self._run_id = run_id
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{mode}_{task_id}_{timestamp}.jsonl"
        self._current_path = os.path.join(self._log_dir, filename)
        self._current_file = open(self._current_path, "w", encoding="utf-8")

        # Log episode header
        self._write_entry({
            "type": "episode_start",
            "timestamp": self._timestamp(),
            "run_id": run_id,
            "task_id": task_id,
            "seed": seed,
            "model": self._redact(model),
            "mode": mode,
        })

    def log_step(self, step: int, seed: int,
                 model_input: Any = None,
                 model_output_raw: str = "",
                 parsed_actions: Any = None,
                 tool_results: Any = None,
                 errors: Any = None,
                 state_hash: str = "",
                 state: Any = None,
                 discrepancies: Any = None,
                 reward: Any = None,
                 tokens: int = 0,
                 latency_ms: float = 0):
        """Log a single step (one model reply)."""
        state_data = state if isinstance(state, dict) else {"hash": state_hash}
        if "hash" not in state_data:
            state_data["hash"] = state_hash

        suspicious_list = []
        if discrepancies:
            for d in discrepancies:
                msg = d.get("message") if isinstance(d, dict) else str(d)
                cat = d.get("category", "discrepancy") if isinstance(d, dict) else "discrepancy"
                suspicious_list.append(f"[{cat}] {msg}")
        if errors:
            for err in errors:
                err_str = str(err)
                if any(kw in err_str.lower() for kw in ["invariant", "mutation", "unauthorized", "deadlock", "corrupt"]):
                    suspicious_list.append(f"[error] {err_str}")

        entry = {
            "type": "step",
            "timestamp": self._timestamp(),
            "step": step,
            "seed": seed,
            "model_input": self._truncate(self._redact_obj(model_input), 3000),
            "model_output_raw": self._truncate(self._redact(str(model_output_raw)), 3000),
            "parsed_actions": parsed_actions or [],
            "tool_result": self._truncate_list(tool_results, 1500),
            "errors": errors or [],
            "suspicious": suspicious_list,
            "discrepancies": discrepancies or [],
            "state": state_data,
            "reward": reward or {},
            "tokens": tokens,
            "latency_ms": round(latency_ms, 2),
        }
        self._write_entry(entry)
        self._print_step(entry)

    def log_baseline_step(self, step: int, tool_name: str, args: dict,
                          result: str, state_hash: str, seed: int = 42,
                          state: Any = None, reward: Any = None,
                          discrepancies: Any = None, errors: Any = None):
        """Log a baseline agent step adhering to the live-run step schema."""
        state_data = state if isinstance(state, dict) else {"hash": state_hash}
        if "hash" not in state_data:
            state_data["hash"] = state_hash

        suspicious_list = []
        if discrepancies:
            for d in discrepancies:
                msg = d.get("message") if isinstance(d, dict) else str(d)
                cat = d.get("category", "discrepancy") if isinstance(d, dict) else "discrepancy"
                suspicious_list.append(f"[{cat}] {msg}")

        entry = {
            "type": "step",
            "timestamp": self._timestamp(),
            "step": step,
            "seed": seed,
            "model_input": f"Baseline action: {tool_name}({args})",
            "model_output_raw": f"{tool_name}({args})",
            "parsed_actions": [{"tool": tool_name, "args": args}],
            "tool_result": [self._truncate(result, 500)],
            "errors": errors or [],
            "suspicious": suspicious_list,
            "discrepancies": discrepancies or [],
            "state": state_data,
            "reward": reward or {},
            "tokens": 0,
            "latency_ms": 0.0,
        }
        self._write_entry(entry)

    def log_verification(self, result: Dict):
        """Log the verifier's output."""
        entry = {
            "type": "verification",
            "timestamp": self._timestamp(),
            **result,
        }
        self._write_entry(entry)
        self._print_verification(result)

    def log_gate_result(self, agent_type: str, task_id: str, result: Dict):
        """Log a baseline gate check result."""
        entry = {
            "type": "gate_check",
            "timestamp": self._timestamp(),
            "agent_type": agent_type,
            "task_id": task_id,
            **result,
        }
        self._write_entry(entry)

    def log_error(self, error: str, context: Dict = None):
        """Log an error event."""
        entry = {
            "type": "error",
            "timestamp": self._timestamp(),
            "error": self._redact(str(error)),
            "context": context or {},
        }
        self._write_entry(entry)

    def log_security_event(self, event_type: str, details: Dict):
        """Log a security-relevant event."""
        entry = {
            "type": "security_event",
            "timestamp": self._timestamp(),
            "event_type": event_type,
            "details": details,
        }
        self._write_entry(entry)
        msg = f"  SECURITY: {event_type} - {json.dumps(details)}"
        print(colorize(msg, "yellow", bold=True))

    def end_episode(self, termination_reason: str, final_score: float,
                    total_steps: int, final_state_hash: str):
        """End the episode logging with normalized termination reason (success, failure, step_cap, time_cap, crash)."""
        norm_reason = str(termination_reason).lower().strip()
        if norm_reason in ["success", "passed"]:
            norm_reason = "success"
        elif norm_reason in ["time_cap", "time_limit", "timeout"]:
            norm_reason = "time_cap"
        elif norm_reason in ["step_cap", "max_steps", "step_limit"]:
            norm_reason = "step_cap"
        elif norm_reason in ["crash", "api_crash", "error", "harness_failure"]:
            norm_reason = "crash"
        else:
            norm_reason = "success" if final_score >= 0.95 else "failure"

        entry = {
            "type": "episode_end",
            "timestamp": self._timestamp(),
            "termination_reason": norm_reason,
            "final_score": final_score,
            "total_steps": total_steps,
            "final_state_hash": final_state_hash,
        }
        self._write_entry(entry)
        if self._current_file:
            self._current_file.close()
            self._current_file = None
        self._print_episode_end(entry)

    def _write_entry(self, entry: Dict):
        """Write a single JSON line."""
        if self._current_file:
            line = json.dumps(entry, default=str, ensure_ascii=False)
            self._current_file.write(line + "\n")
            self._current_file.flush()

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _redact(self, text: str) -> str:
        """Redact potential secrets from text."""
        if not isinstance(text, str):
            return str(text)
        # Redact API key patterns
        text = re.sub(r'(sk-[a-zA-Z0-9]{20,})', '[REDACTED_KEY]', text)
        text = re.sub(r'(api[_-]?key["\s:=]+)["\']?([^"\'\\s]{10,})', r'\1[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'(bearer\s+)\S{20,}', r'\1[REDACTED]', text, flags=re.IGNORECASE)
        return text

    def _redact_obj(self, obj: Any) -> Any:
        """Redact secrets from nested objects."""
        if isinstance(obj, str):
            return self._redact(obj)
        elif isinstance(obj, dict):
            return {k: self._redact_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._redact_obj(item) for item in obj]
        return obj

    def _truncate(self, text: Any, max_len: int = 2000) -> str:
        """Truncate text to max_len."""
        s = str(text) if text is not None else ""
        if len(s) > max_len:
            return s[:max_len] + "...(truncated)"
        return s

    def _truncate_list(self, items: Any, max_item_len: int = 1000) -> list:
        """Truncate items in a list."""
        if not isinstance(items, list):
            return [self._truncate(items, max_item_len)] if items else []
        return [self._truncate(item, max_item_len) for item in items]

    def _print_step(self, entry: Dict):
        """Print readable step info to terminal."""
        actions = entry.get("parsed_actions", [])
        action_str = ", ".join(
            [a.get("tool", "?") if isinstance(a, dict) else str(a) for a in actions]
        ) if actions else "(none)"
        reward = entry.get("reward", {})
        print(f"  Step {entry['step']}: actions=[{action_str}] "
              f"state={entry.get('state', {}).get('hash', '?')} "
              f"reward={reward}")

    def _print_verification(self, result: Dict):
        """Print verification result."""
        score = result.get("score", 0)
        passed = result.get("passed", False)
        status = colorize("PASSED", "green", bold=True) if passed else colorize("FAILED", "red", bold=True)

        print(f"  Verification: {status} (score={score:.4f})")

    def _print_episode_end(self, entry: Dict):
        """Print episode end summary."""
        print(f"  Episode ended: reason={entry['termination_reason']} "
              f"score={entry['final_score']:.4f} steps={entry['total_steps']}")

    @property
    def current_log_path(self) -> Optional[str]:
        return self._current_path
