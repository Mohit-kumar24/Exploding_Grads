"""
KingdomForge Harness Runner

Main entry point. Orchestrates:
1. Baseline checks (reference + no-op)
2. Model evaluation
3. Logging, gate checks, and result collection

Environment variables:
    MODEL       -- Model name (e.g., gpt-luna)
    API_KEY     -- API key
    BASE_URL    -- Base URL for OpenAI-compatible API
    TASK_ID     -- Which task to run (default: demo task)
    SEED        -- Optional seed (default: 42)
    APPEND_FILE -- Instructions file (default: inputs/append.txt)
    MODE        -- full, all, baselines
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from dotenv import load_dotenv
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass


from src.env.database_env import KingdomEnv
from src.env.tasks import get_task, get_all_task_ids, DEMO_TASK_ID
from src.env.tools import get_tool_schemas
from src.verifier.verifier import Verifier
from src.agents.reference_agent import ReferenceAgent
from src.agents.noop_agent import NoopAgent
from src.harness.logger import Logger
from src.harness.terminal import print_red, print_green, print_blue, print_yellow, print_cyan, colorize, cprint



# ---- Exit codes ----
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_GATE_FAILURE = 2
EXIT_CONFIG_ERROR = 3

# ---- Constants from spec ----
MAX_SLOT_TIME_SECONDS = 960   # 16 minutes hard limit for entire live slot
CLEAN_BUILD_LIMIT = 180       # 3 minutes clean docker build
MAX_STEPS = 40
MAX_TIME_SECONDS = 600        # 10 minutes model run
MAX_TOOL_CALLS_PER_REPLY = 10
TOOL_CALL_TIMEOUT = 60        # 60 seconds per tool call
BASELINE_TIME_LIMIT = 120     # 2 minutes for both baselines combined
VERIFIER_TIME_LIMIT = 120     # 2 minutes for final scoring
MAX_CONTEXT_TOKENS = 60000    # Target 64k window with margin
MAX_CONSECUTIVE_API_ERRORS = 3


def load_append_file() -> str:
    """
    Load the append file specified by APPEND_FILE env var according to Section 05.3.
    If APPEND_FILE is explicitly set:
      - If the named file is missing or empty, stops the run with EXIT_CONFIG_ERROR.
    If APPEND_FILE is not set:
      - Checks default 'inputs/append.txt'. If present and non-empty, loads it.
      - If absent, returns empty string (preserving local testing behavior when unset).
    """
    if "APPEND_FILE" in os.environ:
        append_path = os.environ["APPEND_FILE"].strip()
        if not append_path:
            print_red("FATAL: APPEND_FILE environment variable is set to empty path. Run stopped.")
            sys.exit(EXIT_CONFIG_ERROR)
        if not os.path.exists(append_path):
            print_red("FATAL: Append file '%s' not found. Run stopped." % append_path)
            sys.exit(EXIT_CONFIG_ERROR)
        with open(append_path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            print_red("FATAL: Append file '%s' is empty. Run stopped." % append_path)
            sys.exit(EXIT_CONFIG_ERROR)
        return content

    # Default fallback when APPEND_FILE is unset
    default_path = "inputs/append.txt"
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content if content.strip() else ""
    return ""


def _load_append_safe() -> str:
    """Load append file safely. Maintained for backward compatibility with existing tests."""
    return load_append_file()


def execute_tool_call(env, tool_name: str, args: Dict, timeout: float = TOOL_CALL_TIMEOUT):
    """
    Execute a single tool call with timeout enforcement and state isolation.
    Uses non-blocking shutdown so a stalled operation never keeps the harness waiting.
    Operates on a state snapshot so timed-out operations cannot corrupt live state.
    """
    import copy
    import random

    snapshot_env = env.__class__()
    snapshot_env._state = copy.deepcopy(env._state)
    snapshot_env._step_count = env._step_count
    snapshot_env._max_steps = getattr(env, "_max_steps", 40)
    snapshot_env._turn = env._turn
    snapshot_env._done = env._done
    snapshot_env._termination_reason = env._termination_reason
    snapshot_env._event_log = list(env._event_log)
    snapshot_env._seed = env._seed
    snapshot_env._task = env._task
    if env._rng:
        snapshot_env._rng = random.Random()
        snapshot_env._rng.setstate(env._rng.getstate())

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(snapshot_env.step, tool_name, args)
    try:
        result_str, done, info = future.result(timeout=timeout)
        # Commit updated state to live environment
        env._state = snapshot_env._state
        env._step_count = snapshot_env._step_count
        env._max_steps = snapshot_env._max_steps
        env._turn = snapshot_env._turn
        env._done = snapshot_env._done
        env._termination_reason = snapshot_env._termination_reason
        env._event_log = snapshot_env._event_log
        if snapshot_env._rng and env._rng:
            env._rng.setstate(snapshot_env._rng.getstate())
        executor.shutdown(wait=False, cancel_futures=True)
        return result_str, done, info
    except concurrent.futures.TimeoutError:
        # Non-blocking shutdown: do NOT wait for stalled worker thread
        executor.shutdown(wait=False, cancel_futures=True)
        # Live env._state remains untouched; snapshot_env is abandoned
        return ("Error: Tool execution timed out after %ds." % int(timeout), False, {"error": "timeout"})
    except Exception as e:
        executor.shutdown(wait=False, cancel_futures=True)
        return ("Error: Tool execution failed: %s" % str(e), False, {"error": str(e)})


def run_baseline(env, task_id, seed, agent_type, verifier, logger, timeout: float = BASELINE_TIME_LIMIT):
    """
    Run a baseline agent (reference or noop) with active timeout enforcement during execution.
    If execution stalls, timeout unblocks immediately, records failure, and prevents gate from hanging.
    """
    task = get_task(task_id)

    if agent_type == "reference":
        agent = ReferenceAgent()
        mode_label = "REFERENCE"
    else:
        agent = NoopAgent()
        mode_label = "NO-OP"

    print_blue("\n  Running %s baseline for %s (seed=%d)..." % (mode_label, task_id, seed))
    logger.start_episode(
        run_id="baseline_%s_%s" % (agent_type, task_id),
        task_id=task_id, seed=seed, model=agent_type, mode="baseline_%s" % agent_type
    )

    start_time = time.monotonic()

    # Active timeout enforcement during execution
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(agent.run_episode, env, task_id, seed)

    try:
        episode_result = future.result(timeout=timeout)
        executor.shutdown(wait=False, cancel_futures=True)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        elapsed = time.monotonic() - start_time
        print_red("  Baseline exceeded time limit (%ds). FAILED." % int(timeout))
        result = {
            "score": 0.0,
            "passed": False,
            "gate_check": {
                "passed": False,
                "threshold": ">= 0.95" if agent_type == "reference" else "<= 0.05",
                "actual": 0.0,
                "time_limit_exceeded": True,
                "error": "Baseline exceeded time limit (%ds)" % int(timeout),
            },
            "details": {"error": "time_limit_exceeded"},
        }
        logger.end_episode(
            termination_reason="time_cap",
            final_score=0.0,
            total_steps=0,
            final_state_hash=env.state_hash() if getattr(env, '_state', None) else "",
        )
        print_red("  %s baseline FAILED: timed out after %.2fs" % (mode_label, elapsed))
        return result

    # Log steps
    for i, step_data in enumerate(episode_result.get("trajectory", [])):
        logger.log_baseline_step(
            step=i + 1,
            tool_name=step_data.get("tool", ""),
            args=step_data.get("args", {}),
            result=step_data.get("result", ""),
            state_hash=step_data.get("state_hash", ""),
            seed=seed,
            state={"hash": step_data.get("state_hash", "")},
        )

    # Verify
    result = verifier.verify_baseline(env, task, agent_type)
    elapsed = time.monotonic() - start_time

    if elapsed > timeout:
        print_red("  Baseline exceeded time limit (%.2fs > %ds). FAILED." % (elapsed, int(timeout)))
        if "gate_check" in result:
            result["gate_check"]["passed"] = False
            result["gate_check"]["time_limit_exceeded"] = True

    gate = result.get("gate_check", {})
    passed = gate.get("passed", False)
    if gate.get("time_limit_exceeded"):
        term_reason = "time_cap"
    elif passed:
        term_reason = "success"
    else:
        term_reason = "failure"

    result["termination_reason"] = term_reason
    logger.log_verification(result)
    logger.end_episode(
        termination_reason=term_reason,
        final_score=result["score"],
        total_steps=episode_result.get("steps", 0),
        final_state_hash=env.state_hash() if getattr(env, '_state', None) else "",
    )

    if passed:
        print_green("  %s baseline PASSED: score=%.4f (threshold=%s)" % (
            mode_label, result["score"], gate.get("threshold", "?")))
    else:
        print_red("  %s baseline FAILED: score=%.4f (threshold=%s)" % (
            mode_label, result["score"], gate.get("threshold", "?")))

    print_blue("  Baseline completed in %.2fs, %d steps" % (elapsed, episode_result.get("steps", 0)))

    return result


def run_model_episode(env, task_id, seed, model_client, logger, append_text="", slot_start_time: Optional[float] = None):
    """Run a model evaluation episode."""
    task = get_task(task_id)
    tools = get_tool_schemas()

    safe_base = logger._redact(model_client.base_url) if getattr(model_client, "base_url", None) else "(default)"
    print_blue("\n  ============================================================")
    print_blue("  Running Model Evaluation Episode:")
    print("    Task: %s" % task_id)
    print("    Seed: %d" % seed)
    print("    Model: %s" % model_client.model)
    print("    Endpoint: %s" % safe_base)
    print_blue("  ============================================================")

    run_id = "model_%s_%s" % (task_id, uuid.uuid4().hex[:8])
    logger.start_episode(
        run_id=run_id, task_id=task_id, seed=seed,
        model=model_client.model, mode="model"
    )

    # Reset environment
    obs = env.reset(task_id, seed)
    if hasattr(env, "set_max_steps"):
        env.set_max_steps(MAX_STEPS * MAX_TOOL_CALLS_PER_REPLY)

    # Build system prompt
    system_prompt = (
        "You are an AI agent managing a medieval kingdom in the KingdomForge simulation. "
        "You must achieve the task objectives by using the available tools.\n\n"
        "TASK: %s\n\n"
        "CURRENT KINGDOM STATE:\n%s\n\n"
        "PLANNING AND EXECUTION RULES:\n"
        "1. Inspect initial state and objectives upfront.\n"
        "2. Upfront resource budgeting: Calculate total gold, stone, and wood required for ALL mandatory objectives before spending. Gold is non-renewable unless a market is operating. Do not spend scarce gold on optional buildings.\n"
        "3. Check population capacity in the kingdom state before building houses; if existing capacity already satisfies the target, building houses is unnecessary.\n"
        "4. Action batching: You can issue multiple compatible tool calls in a single reply (up to %d per reply). Batch independent actions together.\n"
        "5. Avoid redundant queries: The kingdom state updates after actions. Do not repeatedly call view_kingdom when no state has changed.\n"
        "6. Advance turns purposefully: Advance turns only when waiting for assigned workers to produce resources. Advancing turns consumes food and will NOT produce gold without a market.\n"
        "7. Dynamic error recovery: If an action returns an error (e.g. 'Not enough gold'), do not loop advance_turn. Re-check resources and adapt your plan immediately.\n"
        "8. Completion: Confirm all objectives are met with check_objectives, then immediately call submit_answer(answer='done') to conclude.\n"
        "- Maximum model replies: %d.\n"
    ) % (task.instruction, json.dumps(obs.get("kingdom", {}), indent=2), MAX_TOOL_CALLS_PER_REPLY, MAX_STEPS)

    if append_text:
        system_prompt += "\n\nADDITIONAL INSTRUCTIONS:\n%s\n" % append_text

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": "Begin managing the kingdom. Check the current state and work toward the objectives."})

    model_start_time = time.monotonic()
    step = 0
    consecutive_api_errors = 0
    episode_crashed = False
    crash_error_msg = ""

    while step < MAX_STEPS:
        # Check slot-level 16-minute hard stop
        if slot_start_time is not None:
            slot_elapsed = time.monotonic() - slot_start_time
            if slot_elapsed >= MAX_SLOT_TIME_SECONDS:
                print_red("  16-minute live slot hard stop reached (%.0fs >= %ds). Ending episode." % (
                    slot_elapsed, MAX_SLOT_TIME_SECONDS))
                env._done = True
                env._termination_reason = "time_cap"
                break

        # Check model episode 10-minute time limit
        elapsed = time.monotonic() - model_start_time
        if elapsed > MAX_TIME_SECONDS:
            print_red("  Model episode time limit reached (%.0fs > %ds). Ending episode." % (
                elapsed, MAX_TIME_SECONDS))
            env._done = True
            env._termination_reason = "time_cap"
            break

        if env.is_done:
            break

        step += 1
        cprint("\n  [Step %d/%d]" % (step, MAX_STEPS), color="blue", bold=True)
        print("    Calling model endpoint...")

        # Call model with monotonic latency measurement
        call_start = time.monotonic()
        response = model_client.chat(messages, tools)
        call_latency = (time.monotonic() - call_start) * 1000

        # Handle errors
        if response.get("error"):
            error_msg = response["error"]
            print_red("    Model API error at step %d: %s" % (step, error_msg))
            consecutive_api_errors += 1
            messages.append({"role": "assistant", "content": "Error occurred."})
            messages.append({"role": "user", "content": "An API error occurred: %s. Please try again." % error_msg})

            logger.log_step(
                step=step, seed=seed,
                model_input=_trim_for_log(messages),
                model_output_raw=response.get("raw", ""),
                errors=[error_msg],
                state_hash=env.state_hash(),
                tokens=response.get("tokens", 0),
                latency_ms=call_latency,
            )

            lower_err = error_msg.lower()
            is_auth_error = any(kw in lower_err for kw in [
                "401", "authentication", "unauthorized", "invalid_api_key",
                "invalid api key", "incorrect api key", "permission_denied",
                "forbidden", "403"
            ])

            if is_auth_error or consecutive_api_errors >= MAX_CONSECUTIVE_API_ERRORS:
                fail_reason = "Authentication failure" if is_auth_error else "Repeated API failures (%d consecutive)" % consecutive_api_errors
                print_red("    Fatal API condition: %s. Terminating episode as crash." % fail_reason)
                episode_crashed = True
                crash_error_msg = error_msg
                break

            continue

        consecutive_api_errors = 0

        content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])
        tokens_used = response.get("tokens", 0)

        print("    Model replied in %.1fms (tokens: %d)" % (call_latency, tokens_used))

        if content and str(content).strip():
            print_cyan("    Model thought / text:")
            for line in str(content).strip().splitlines()[:15]:
                print("      %s" % line)
            if len(str(content).strip().splitlines()) > 15:
                print("      ...(remaining lines trimmed for display)")

        # Process tool calls
        tool_results = []
        parsed_actions = []

        if tool_calls:
            print_yellow("    Tool calls requested: %d" % len(tool_calls), bold=True)
            # Limit to MAX_TOOL_CALLS_PER_REPLY
            for i, tc in enumerate(tool_calls):
                # Check overall model-run time cap
                if time.monotonic() - model_start_time > MAX_TIME_SECONDS:
                    env._done = True
                    env._termination_reason = "time_cap"
                    tool_results.append({
                        "tool_call_id": tc.get("id", ""),
                        "content": "Error: Maximum episode time (%ds) exceeded." % MAX_TIME_SECONDS,
                    })
                    break

                if i >= MAX_TOOL_CALLS_PER_REPLY:
                    tool_results.append({
                        "tool_call_id": tc.get("id", ""),
                        "content": "Error: Maximum %d tool calls per reply exceeded." % MAX_TOOL_CALLS_PER_REPLY,
                    })
                    parsed_actions.append({"tool": tc.get("name", "?"), "rejected": True})
                    continue

                tool_name = tc.get("name", "")
                args = tc.get("arguments", {})

                # Handle parse errors in arguments
                if "_parse_error" in args:
                    result_str = "Error: Could not parse arguments: %s" % args["_parse_error"]
                    tool_results.append({
                        "tool_call_id": tc.get("id", ""),
                        "content": result_str,
                    })
                    parsed_actions.append({"tool": tool_name, "error": "parse_error"})
                    print_red("      -> Tool call #%d: %s [Parse Error: %s]" % (i + 1, tool_name, args["_parse_error"]))
                    continue

                # Execute with timeout
                args_preview = json.dumps(args, default=str)
                if len(args_preview) > 100:
                    args_preview = args_preview[:100] + "..."
                print("      -> Tool call #%d: %s(args=%s)" % (i + 1, tool_name, args_preview))

                parsed_actions.append({"tool": tool_name, "args": args})
                result_str, done, info = execute_tool_call(env, tool_name, args, timeout=TOOL_CALL_TIMEOUT)

                tool_results.append({
                    "tool_call_id": tc.get("id", ""),
                    "content": result_str[:2000],
                })

                res_preview = result_str[:120].replace("\n", " ").strip()
                if len(result_str) > 120:
                    res_preview += "..."
                print("         Result: %s" % res_preview)
                print("         State Hash: %s" % env.state_hash())

            # Build assistant message with tool calls
            assistant_msg = {"role": "assistant", "content": content}
            if response.get("tool_calls"):
                assistant_tool_calls = []
                for i, tc in enumerate(response["tool_calls"]):
                    tc_args = tc.get("arguments", {})
                    # Sanitize malformed arguments to valid empty JSON object in assistant history
                    if isinstance(tc_args, dict) and "_parse_error" in tc_args:
                        serialized_args = "{}"
                    elif isinstance(tc_args, dict):
                        serialized_args = json.dumps(tc_args)
                    else:
                        serialized_args = "{}"

                    assistant_tool_calls.append({
                        "id": tc.get("id", "call_%d" % i),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": serialized_args,
                        }
                    })
                assistant_msg["tool_calls"] = assistant_tool_calls
            messages.append(assistant_msg)

            # Add tool results
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"],
                })
        else:
            # No tool calls -- text-only reply
            print_yellow("    (No tool calls made in this step)")
            messages.append({"role": "assistant", "content": content or ""})
            messages.append({
                "role": "user",
                "content": "You did not make any tool calls. Use the available tools to make progress on the objectives. Use check_objectives to see what remains."
            })

        # Compute provisional reward and evaluate discrepancies
        verifier_step = Verifier()
        provisional = verifier_step.verify(env, task)
        step_reward = {
            "score": provisional.get("score", 0.0),
            "breakdown": provisional.get("details", {}).get("objective_breakdown", []),
            "penalties": provisional.get("details", {}).get("penalties", []),
        }
        discrepancies = list(getattr(env, "discrepancies", []))
        if discrepancies:
            for d in discrepancies:
                print_yellow("         [DISCREPANCY DETECTED: %s - %s]" % (d.get("category"), d.get("detail")))

        # Log step
        logger.log_step(
            step=step, seed=seed,
            model_input=_trim_for_log(messages),
            model_output_raw=response.get("raw", "")[:1000],
            parsed_actions=parsed_actions,
            tool_results=[tr["content"][:500] for tr in tool_results],
            errors=[response["error"]] if response.get("error") else [],
            state_hash=env.state_hash(),
            state={"hash": env.state_hash(), "turn": getattr(env, "turn", 0), "resources": dict(env.state.get("resources", {})) if hasattr(env, "state") else {}},
            discrepancies=discrepancies,
            reward=step_reward,
            tokens=response.get("tokens", 0),
            latency_ms=call_latency,
        )

        # Trim context if too long
        messages = _trim_context(messages, system_prompt)

    # Episode ended -- set termination_reason if step limit reached
    if not env.termination_reason and step >= MAX_STEPS:
        env._done = True
        env._terminated = True
        env._termination_reason = "step_cap"

    # Verify
    verifier = Verifier()
    result = verifier.verify(env, task)
    result["steps_used"] = step
    if env.termination_reason:
        result["termination_reason"] = env.termination_reason
    if "details" in result and isinstance(result["details"], dict):
        result["details"]["tool_calls_used"] = env.step_count

    if episode_crashed:
        # Override score and termination reason to reflect crash, not legitimate completion
        result["score"] = 0.0
        result["passed"] = False
        result["termination_reason"] = "crash"
        result["details"]["api_crash"] = crash_error_msg

        logger.log_verification(result)
        logger.end_episode(
            termination_reason="crash",
            final_score=0.0,
            total_steps=step,
            final_state_hash=env.state_hash(),
        )
        total_elapsed = time.monotonic() - model_start_time
        print_red("\n  Model episode crashed: score=0.0000, steps=%d, time=%.1fs, reason=crash" % (
            step, total_elapsed), bold=True)
        return result

    result["termination_reason"] = env.termination_reason or "step_cap"
    result["outcome"] = "success" if result.get("passed") else "failure"
    logger.log_verification(result)
    logger.end_episode(
        termination_reason=env.termination_reason or "step_cap",
        final_score=result["score"],
        total_steps=step,
        final_state_hash=env.state_hash(),
    )

    total_elapsed = time.monotonic() - model_start_time
    status_str = colorize("PASSED", "green", bold=True) if result.get("passed") else colorize("FAILED", "red", bold=True)
    print_green("\n  ============================================================", bold=True)
    print_green("  Model Episode Summary:", bold=True)
    print("    Task: %s" % task_id)
    print("    Verdict: %s" % status_str)
    print("    Final Score: %.4f" % result["score"])
    print("    Termination Reason: %s" % (env.termination_reason or "step_cap"))
    print("    Total Steps: %d" % step)
    print("    Duration: %.1fs" % total_elapsed)
    print_green("  ============================================================", bold=True)

    return result


def _trim_for_log(messages, max_chars=3000):

    """Trim messages for logging to prevent huge logs."""
    trimmed = []
    for m in messages[-5:]:  # Last 5 messages
        content = m.get("content", "")
        if isinstance(content, str) and len(content) > max_chars:
            content = content[:max_chars] + "...(trimmed)"
        trimmed.append({"role": m.get("role", "?"), "content": content})
    return trimmed


def _trim_context(messages, system_prompt, max_messages=30):
    """Trim conversation history to fit context window while preserving OpenAI/Azure tool-call protocol.

    Rules for valid OpenAI/Azure conversation history:
    1. System prompt (index 0) and initial user task prompt (index 1) are preserved.
    2. Every role: 'tool' message must be immediately preceded by an assistant message
       containing a matching tool_call id in assistant.tool_calls.
    3. An assistant message with tool_calls must have all its tool calls answered by tool messages.
    4. Slicing must never start in the middle of a tool call sequence.
    """
    if len(messages) <= max_messages:
        return messages

    head = [messages[0]]  # system
    if len(messages) > 1:
        head.append(messages[1])  # first user message

    # Target number of recent messages to keep
    target_tail_len = max(2, max_messages - len(head) - 1)
    start_idx = max(len(head), len(messages) - target_tail_len)

    # Never start tail on an orphaned 'tool' response
    while start_idx < len(messages) and messages[start_idx].get("role") == "tool":
        start_idx += 1

    tail = list(messages[start_idx:])

    # Validate and clean turns so tool_calls and tool results stay paired
    cleaned_tail = []
    i = 0
    while i < len(tail):
        msg = tail[i]
        if msg.get("role") == "tool":
            # Orphaned tool message without preceding assistant
            i += 1
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected_ids = {tc.get("id") for tc in msg.get("tool_calls", []) if tc.get("id")}
            j = i + 1
            found_ids = set()
            tool_msgs = []
            while j < len(tail) and tail[j].get("role") == "tool":
                t_id = tail[j].get("tool_call_id")
                if t_id in expected_ids:
                    found_ids.add(t_id)
                    tool_msgs.append(tail[j])
                j += 1
            if expected_ids == found_ids:
                cleaned_tail.append(msg)
                cleaned_tail.extend(tool_msgs)
            i = j
        else:
            cleaned_tail.append(msg)
            i += 1

    dropped_count = len(messages) - (len(head) + len(cleaned_tail))
    trimmed_result = list(head)
    if dropped_count > 0:
        trimmed_result.append({
            "role": "user",
            "content": "[Context trimmed: %d earlier messages removed to fit window. Continue working toward objectives.]" % dropped_count
        })
    trimmed_result.extend(cleaned_tail)

    return trimmed_result


def run_baselines_for_task(task_id, seed, logger, combined_timeout: float = BASELINE_TIME_LIMIT):
    """Run both baselines for a single task within the combined 2-minute limit."""
    verifier = Verifier()
    combined_start = time.monotonic()

    env = KingdomEnv()
    ref_result = run_baseline(env, task_id, seed, "reference", verifier, logger, timeout=combined_timeout)
    env.close()

    elapsed_ref = time.monotonic() - combined_start
    remaining_noop_time = max(0.0, combined_timeout - elapsed_ref)

    if remaining_noop_time <= 0 or not ref_result.get("gate_check", {}).get("passed", False):
        noop_result = {
            "score": 0.0,
            "passed": False,
            "gate_check": {
                "passed": False,
                "threshold": "<= 0.05",
                "actual": 0.0,
                "time_limit_exceeded": remaining_noop_time <= 0,
                "error": "Combined baselines exceeded 2-minute time limit (%ds)" % int(combined_timeout) if remaining_noop_time <= 0 else "Skipped due to reference failure",
            },
            "details": {"error": "combined_time_limit_exceeded" if remaining_noop_time <= 0 else "ref_failed"},
        }
    else:
        env = KingdomEnv()
        noop_result = run_baseline(env, task_id, seed, "noop", verifier, logger, timeout=remaining_noop_time)
        env.close()

    combined_elapsed = time.monotonic() - combined_start
    if combined_elapsed > combined_timeout:
        print_red("  Combined baselines exceeded time limit (%.2fs > %ds). FAILED." % (combined_elapsed, int(combined_timeout)))
        if "gate_check" in ref_result:
            ref_result["gate_check"]["passed"] = False
            ref_result["gate_check"]["time_limit_exceeded"] = True
        if "gate_check" in noop_result:
            noop_result["gate_check"]["passed"] = False
            noop_result["gate_check"]["time_limit_exceeded"] = True

    return ref_result, noop_result


def main(args_list=None):
    """Main entry point."""
    print("=" * 60)
    print("  KingdomForge -- RL Environment Harness")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="KingdomForge Harness Runner")
    parser.add_argument("--task", "-t", default=os.environ.get("TASK_ID", DEMO_TASK_ID).strip(), help="Task ID to evaluate")
    parser.add_argument("--seed", "-s", type=int, default=int(os.environ.get("SEED", "42")), help="Random seed")
    parser.add_argument("--mode", "-m", default=os.environ.get("MODE", "full").lower().strip(), help="Execution mode (full, all, baselines)")
    parser.add_argument("--model", default=os.environ.get("MODEL", "").strip(), help="Model name")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "").strip(), help="API key")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "").strip(), help="Base URL")
    args, unknown = parser.parse_known_args(args_list)

    mode = args.mode.lower().strip()
    task_id = args.task.strip()
    seed = args.seed
    model_name = args.model.strip()
    api_key = args.api_key.strip()
    base_url = args.base_url.strip()

    print_blue("\n  Configuration:")
    print("    MODE=%s" % mode)
    print("    TASK_ID=%s" % task_id)
    print("    SEED=%d" % seed)
    print("    MODEL=%s" % (model_name if model_name else "(none)"))
    print("    BASE_URL=%s" % (base_url if base_url else "(none)"))

    logger = Logger(log_dir="./logs")
    gate_results = {}
    slot_start_time = time.monotonic()

    if mode == "baselines":
        # Run baselines for all tasks
        print_blue("\n  MODE=baselines: Running baselines for ALL tasks...")
        all_tasks = get_all_task_ids()
        all_passed = True

        for tid in all_tasks:
            ref_result, noop_result = run_baselines_for_task(tid, seed, logger)
            ref_gate = ref_result.get("gate_check", {}).get("passed", False)
            noop_gate = noop_result.get("gate_check", {}).get("passed", False)

            gate_results[tid] = {
                "reference": {"score": ref_result["score"], "passed": ref_gate},
                "noop": {"score": noop_result["score"], "passed": noop_gate},
                "gate_passed": ref_gate and noop_gate,
            }

            if not (ref_gate and noop_gate):
                all_passed = False

        # Write gate.json
        _write_gate_json(gate_results)

        if all_passed:
            print_green("\n  ALL baseline gates PASSED.", bold=True)
            sys.exit(EXIT_SUCCESS)
        else:
            print_red("\n  Some baseline gates FAILED. See gate.json for details.", bold=True)
            sys.exit(EXIT_GATE_FAILURE)

    elif mode == "all":
        # Run baselines + model for all tasks
        print_blue("\n  MODE=all: Running baselines + model for ALL tasks...")
        append_text = load_append_file()
        all_tasks = get_all_task_ids()
        all_passed = True

        for tid in all_tasks:
            ref_result, noop_result = run_baselines_for_task(tid, seed, logger)
            ref_gate = ref_result.get("gate_check", {}).get("passed", False)
            noop_gate = noop_result.get("gate_check", {}).get("passed", False)

            gate_results[tid] = {
                "reference": {"score": ref_result["score"], "passed": ref_gate},
                "noop": {"score": noop_result["score"], "passed": noop_gate},
                "gate_passed": ref_gate and noop_gate,
            }

            if not (ref_gate and noop_gate):
                print_red("  Baseline gate FAILED for %s. Skipping model." % tid)
                all_passed = False
                continue

            # Run model
            if model_name.lower() == "mock":
                from src.harness.model_client import MockModelClient
                client = MockModelClient()
            elif not model_name or not api_key:
                print_red("  No MODEL/API_KEY set. Skipping model run for %s." % tid)
                continue
            else:
                from src.harness.model_client import ModelClient
                client = ModelClient(model_name, api_key, base_url)
            env = KingdomEnv()
            run_model_episode(env, tid, seed, client, logger, append_text, slot_start_time=slot_start_time)
            env.close()

        _write_gate_json(gate_results)
        if all_passed:
            sys.exit(EXIT_SUCCESS)
        else:
            sys.exit(EXIT_GATE_FAILURE)

    elif mode == "full":
        # Default: run baselines for selected task, then model
        print_blue("\n  MODE=full: Running for task=%s..." % task_id)

        # Validate task exists
        try:
            get_task(task_id)
        except ValueError as e:
            print_red("  FATAL: %s" % str(e))
            sys.exit(EXIT_CONFIG_ERROR)

        # Validate append file early so configuration errors fail immediately
        append_text = load_append_file()

        # Run baselines
        ref_result, noop_result = run_baselines_for_task(task_id, seed, logger)
        ref_gate = ref_result.get("gate_check", {}).get("passed", False)
        noop_gate = noop_result.get("gate_check", {}).get("passed", False)

        gate_results[task_id] = {
            "reference": {"score": ref_result["score"], "passed": ref_gate},
            "noop": {"score": noop_result["score"], "passed": noop_gate},
            "gate_passed": ref_gate and noop_gate,
        }

        if not ref_gate:
            print_red("\n  FATAL: Reference baseline FAILED (%.4f < 0.95). Model not called." % ref_result["score"], bold=True)
            _write_gate_json(gate_results)
            sys.exit(EXIT_GATE_FAILURE)

        if not noop_gate:
            print_red("\n  FATAL: No-op baseline FAILED (%.4f > 0.05). Model not called." % noop_result["score"], bold=True)
            _write_gate_json(gate_results)
            sys.exit(EXIT_GATE_FAILURE)

        # Check slot time budget before model episode
        slot_elapsed = time.monotonic() - slot_start_time
        if slot_elapsed >= MAX_SLOT_TIME_SECONDS:
            print_red("\n  FATAL: 16-minute live slot hard stop reached before model call (%.0fs >= %ds). Model not called." % (
                slot_elapsed, MAX_SLOT_TIME_SECONDS), bold=True)
            _write_gate_json(gate_results)
            sys.exit(EXIT_GATE_FAILURE)

        print_green("\n  Both baselines PASSED. Proceeding to model run.", bold=True)

        # Run model
        if model_name.lower() == "mock":
            from src.harness.model_client import MockModelClient
            client = MockModelClient()
        elif not model_name or not api_key:
            print_red("  No MODEL or API_KEY set. Model run skipped.")
            print_blue("  Set MODEL, API_KEY, BASE_URL in .env to run the model.")
            _write_gate_json(gate_results)
            sys.exit(EXIT_SUCCESS)
        else:
            from src.harness.model_client import ModelClient
            client = ModelClient(model_name, api_key, base_url)

        env = KingdomEnv()
        model_result = run_model_episode(env, task_id, seed, client, logger, append_text, slot_start_time=slot_start_time)
        env.close()

        _write_gate_json(gate_results)
        print_green("\n  Run complete. Logs written to ./logs/")
        if model_result.get("termination_reason") == "crash":
            sys.exit(EXIT_ERROR)
        sys.exit(EXIT_SUCCESS)

    else:
        print_red("  FATAL: Unknown MODE='%s'. Use: full, all, baselines" % mode)
        sys.exit(EXIT_CONFIG_ERROR)


def _write_gate_json(gate_results):
    """Write gate.json to baseline-results/ atomically, merging with existing task results."""
    os.makedirs("baseline-results", exist_ok=True)
    path = os.path.join("baseline-results", "gate.json")
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(gate_results)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp_path, path)
    print_blue("  Gate results written to %s" % path)


if __name__ == "__main__":
    main()
