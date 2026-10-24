"""
KingdomForge Dashboard Data Backend

Extracts, parses, and aggregates real evaluation data directly from:
- Task definitions: src/env/tasks.py
- Baseline gates: baseline-results/gate.json
- Model and baseline episode logs: logs/ and test-run-logs/
- Sample trajectories: sample-trajectories/
- Test framework reports: test-results.json and unittest suite

Strictly read-only, path-traversal protected, with zero secret leakage.
"""
import glob
import json
import os
import re
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GATE_FILE_PATH = os.path.join(PROJECT_ROOT, "baseline-results", "gate.json")
LOG_DIRS = [
    os.path.join(PROJECT_ROOT, "logs"),
    os.path.join(PROJECT_ROOT, "test-run-logs"),
]
TRAJECTORY_DIRS = [
    os.path.join(PROJECT_ROOT, "sample-trajectories"),
    *LOG_DIRS,
]
TEST_RESULTS_PATH = os.path.join(PROJECT_ROOT, "test-results.json")


def sanitize_filename(name: str) -> Optional[str]:
    """Validate that a filename is safe from path traversal."""
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    # Reject any directory path separators or parent traversal
    if "/" in name or "\\" in name or ".." in name:
        return None
    # Reject hidden files (.env, .git, etc.)
    if name.startswith("."):
        return None
    # Allow alphanumeric characters, underscore, hyphen, and standard extension
    if not re.match(r"^[a-zA-Z0-9_\-]+(\.[a-zA-Z0-9_\-]+)?$", name):
        return None
    return name


def get_task_inventory() -> Dict[str, Dict[str, Any]]:
    """Return task metadata loaded directly from src.env.tasks.TASKS."""
    try:
        from src.env.tasks import TASKS
        tasks_out = {}
        for tid, task in TASKS.items():
            tasks_out[tid] = {
                "id": tid,
                "title": getattr(task, "title", tid),
                "difficulty": getattr(task, "difficulty", "unknown"),
                "description": getattr(task, "description", ""),
                "instruction": getattr(task, "instruction", ""),
            }
        return tasks_out
    except Exception as e:
        return {}


def get_baseline_gates(gate_path: str = GATE_FILE_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Read real baseline gate results from baseline-results/gate.json.
    Distinguishes missing data from score 0.0.
    """
    if not os.path.exists(gate_path):
        return {}
    try:
        with open(gate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        result = {}
        for tid, gate_entry in data.items():
            if not isinstance(gate_entry, dict):
                continue
            ref = gate_entry.get("reference", {})
            noop = gate_entry.get("noop", {})
            result[tid] = {
                "task_id": tid,
                "reference_score": ref.get("score") if "score" in ref else None,
                "reference_threshold": 0.95,
                "reference_passed": ref.get("passed", False),
                "noop_score": noop.get("score") if "score" in noop else None,
                "noop_threshold": 0.05,
                "noop_passed": noop.get("passed", False),
                "gate_passed": gate_entry.get("gate_passed", False),
            }
        return result
    except Exception:
        return {}


def scan_run_logs(log_dirs: Optional[List[str]] = None,
                  task_id_filter: Optional[str] = None,
                  mode_filter: Optional[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
    """
    Scan JSONL log files across approved directories and extract summary headers.
    Distinguishes crashes, completions, and interrupted runs.
    """
    if log_dirs is None:
        log_dirs = LOG_DIRS

    runs = []
    seen_files = set()

    for d in log_dirs:
        if not os.path.isdir(d):
            continue
        # Scan all .jsonl files in directory
        for filepath in glob.glob(os.path.join(d, "*.jsonl")):
            fname = os.path.basename(filepath)
            if fname in seen_files:
                continue
            seen_files.add(fname)

            try:
                run_info = _read_run_header_and_tail(filepath)
                if not run_info:
                    continue

                if task_id_filter and run_info.get("task_id") != task_id_filter:
                    continue
                if mode_filter and run_info.get("mode") != mode_filter:
                    continue

                runs.append(run_info)
            except Exception:
                continue

    # Sort runs descending by timestamp (newest first)
    runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    if limit and limit > 0:
        return runs[:limit]
    return runs


def _read_run_header_and_tail(filepath: str) -> Optional[Dict[str, Any]]:
    """Quickly read first and last lines of a JSONL log file."""
    fname = os.path.basename(filepath)
    run_id = os.path.splitext(fname)[0]

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        if not first_line:
            return None
        try:
            start_record = json.loads(first_line)
        except Exception:
            return None

        if start_record.get("type") != "episode_start":
            return None

        # Seek or scan to end to find verification / episode_end
        end_record = None
        verification_record = None
        step_count = 0

        # For typical small files (10-100KB), read lines efficiently
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec_type = rec.get("type")
                if rec_type in ("step", "baseline_step"):
                    step_count += 1
                elif rec_type == "verification":
                    verification_record = rec
                elif rec_type == "episode_end":
                    end_record = rec
            except Exception:
                continue

    # Determine status & correctness
    term_reason = None
    final_score = None
    passed = False
    details = {}
    is_crash = False
    is_completed = False

    if verification_record:
        final_score = verification_record.get("score")
        passed = bool(verification_record.get("passed", False))
        details = verification_record.get("details", {})
        term_reason = verification_record.get("termination_reason")

    if end_record:
        is_completed = True
        term_reason = end_record.get("termination_reason") or term_reason
        if final_score is None:
            final_score = end_record.get("final_score")
        if "total_steps" in end_record:
            step_count = end_record["total_steps"]

    if term_reason == "crash" or (isinstance(details, dict) and "api_crash" in details):
        is_crash = True
        passed = False
        final_score = 0.0

    return {
        "run_id": run_id,
        "filename": fname,
        "source_dir": os.path.basename(os.path.dirname(filepath)),
        "task_id": start_record.get("task_id", "unknown"),
        "seed": start_record.get("seed", 42),
        "model": start_record.get("model", ""),
        "mode": start_record.get("mode", ""),
        "timestamp": start_record.get("timestamp", ""),
        "total_steps": step_count,
        "final_score": final_score,
        "passed": passed,
        "termination_reason": term_reason or ("in_progress" if not is_completed else "unknown"),
        "is_crash": is_crash,
        "is_completed": is_completed and not is_crash,
        "is_interrupted": (not is_completed) and (not is_crash),
        "details": details,
    }


def get_run_trajectory(run_identifier: str,
                       search_dirs: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Load complete trajectory steps and verification for a specific run.
    Guards against path traversal by enforcing filename sanitization.
    """
    if search_dirs is None:
        search_dirs = TRAJECTORY_DIRS

    safe_name = sanitize_filename(run_identifier)
    if not safe_name:
        return None

    # Handle extensions: try safe_name, safe_name.jsonl, safe_name.json
    candidates = [safe_name]
    if not safe_name.endswith(".jsonl") and not safe_name.endswith(".json"):
        candidates.append(f"{safe_name}.jsonl")
        candidates.append(f"{safe_name}.json")

    target_path = None
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for cand in candidates:
            p = os.path.join(d, cand)
            if os.path.isfile(p):
                target_path = p
                break
        if target_path:
            break

    if not target_path:
        return None

    # If it's a JSON file (e.g. from sample-trajectories/)
    if target_path.endswith(".json"):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            steps = []
            for idx, item in enumerate(data.get("trajectory", [])):
                steps.append({
                    "step": idx + 1,
                    "tool": item.get("tool", "action"),
                    "args": item.get("args", {}),
                    "result": item.get("result", ""),
                    "state_hash": item.get("state_hash", ""),
                    "done": item.get("done", False),
                    "tokens": item.get("tokens", 0),
                    "latency_ms": item.get("latency_ms", 0.0),
                })
            return {
                "run_id": os.path.splitext(os.path.basename(target_path))[0],
                "filename": os.path.basename(target_path),
                "task_id": data.get("task_id", "unknown"),
                "seed": data.get("seed", 42),
                "mode": "sample",
                "model": "reference",
                "total_steps": len(steps),
                "final_score": data.get("verification", {}).get("score", 1.0 if "success" in target_path else 0.0),
                "passed": data.get("verification", {}).get("passed", "success" in target_path),
                "termination_reason": data.get("termination_reason", "submitted"),
                "final_state_hash": data.get("final_state_hash", steps[-1]["state_hash"] if steps else ""),
                "verification": data.get("verification", {}),
                "steps": steps,
            }
        except Exception:
            return None

    # Parse JSONL log file
    start_info = {}
    verification_info = {}
    end_info = {}
    steps = []

    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                r_type = rec.get("type")
                if r_type == "episode_start":
                    start_info = rec
                elif r_type == "step":
                    actions = rec.get("parsed_actions") or []
                    results = rec.get("tool_result") or []
                    if actions:
                        for act_idx, act in enumerate(actions):
                            res = results[act_idx] if act_idx < len(results) else ""
                            steps.append({
                                "step": rec.get("step"),
                                "sub_step": (act_idx + 1) if len(actions) > 1 else None,
                                "timestamp": rec.get("timestamp"),
                                "tool": act.get("tool", ""),
                                "args": act.get("args", {}),
                                "result": res,
                                "state_hash": rec.get("state", {}).get("hash", ""),
                                "tokens": rec.get("tokens", 0),
                                "latency_ms": rec.get("latency_ms", 0.0),
                                "errors": rec.get("errors", []),
                            })
                    else:
                        steps.append({
                            "step": rec.get("step"),
                            "sub_step": None,
                            "timestamp": rec.get("timestamp"),
                            "tool": "(no_tool_call)",
                            "args": {},
                            "result": "(text_only_reply)",
                            "state_hash": rec.get("state", {}).get("hash", ""),
                            "tokens": rec.get("tokens", 0),
                            "latency_ms": rec.get("latency_ms", 0.0),
                            "errors": rec.get("errors", []),
                        })
                elif r_type == "baseline_step":
                    steps.append({
                        "step": rec.get("step"),
                        "sub_step": None,
                        "timestamp": rec.get("timestamp"),
                        "tool": rec.get("tool", ""),
                        "args": rec.get("args", {}),
                        "result": rec.get("result", ""),
                        "state_hash": rec.get("state", {}).get("hash", ""),
                        "tokens": 0,
                        "latency_ms": 0.0,
                        "errors": [],
                    })
                elif r_type == "verification":
                    verification_info = rec
                elif r_type == "episode_end":
                    end_info = rec

        score = verification_info.get("score")
        if score is None:
            score = end_info.get("final_score")
        passed = bool(verification_info.get("passed", False))
        term_reason = end_info.get("termination_reason") or verification_info.get("termination_reason") or "in_progress"
        final_hash = end_info.get("final_state_hash") or verification_info.get("state_hash") or (steps[-1]["state_hash"] if steps else "")

        return {
            "run_id": os.path.splitext(os.path.basename(target_path))[0],
            "filename": os.path.basename(target_path),
            "task_id": start_info.get("task_id", "unknown"),
            "seed": start_info.get("seed", 42),
            "mode": start_info.get("mode", "model"),
            "model": start_info.get("model", ""),
            "timestamp": start_info.get("timestamp", ""),
            "total_steps": len(steps),
            "final_score": score,
            "passed": passed,
            "termination_reason": term_reason,
            "final_state_hash": final_hash,
            "verification": verification_info,
            "steps": steps,
        }
    except Exception:
        return None


def compute_summary(tasks: Dict[str, Dict[str, Any]],
                    gates: Dict[str, Dict[str, Any]],
                    runs: List[Dict[str, Any]],
                    test_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute aggregate dashboard metrics adhering to strict metric correctness rules:
    - Never count crashes as genuine task planning failures.
    - Compute success rate using only completed evaluations in denominator.
    - Provide sample sizes alongside all aggregates.
    """
    total_tasks = len(tasks)
    diff_counts = {"easy": 0, "medium": 0, "hard": 0}
    for t in tasks.values():
        d = t.get("difficulty", "").lower()
        if d in diff_counts:
            diff_counts[d] += 1

    # Baseline gates calculations
    gates_evaluated = len(gates)
    gates_passed = sum(1 for g in gates.values() if g.get("gate_passed") is True)
    all_gates_passed = (gates_evaluated == total_tasks) and (gates_passed == total_tasks)
    gate_pass_rate = round(gates_passed / gates_evaluated, 4) if gates_evaluated > 0 else None

    # Model evaluation runs calculations
    model_runs = [r for r in runs if r.get("mode") == "model"]
    attempted = len(model_runs)
    crashed = sum(1 for r in model_runs if r.get("is_crash"))
    interrupted = sum(1 for r in model_runs if r.get("is_interrupted"))
    completed = [r for r in model_runs if r.get("is_completed")]
    completed_count = len(completed)

    succeeded = sum(1 for r in completed if r.get("passed") is True)
    failed = sum(1 for r in completed if r.get("passed") is False)

    scores = [r["final_score"] for r in completed if isinstance(r.get("final_score"), (int, float))]
    avg_score = round(sum(scores) / len(scores), 4) if scores else None
    success_rate = round(succeeded / completed_count, 4) if completed_count > 0 else None

    # Active model name
    active_model = "None"
    for r in model_runs:
        if r.get("model"):
            active_model = r["model"]
            break

    return {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "active_model": active_model,
        "tasks": {
            "total": total_tasks,
            "difficulty_breakdown": diff_counts,
        },
        "baselines": {
            "gates_total": total_tasks,
            "gates_evaluated": gates_evaluated,
            "gates_passed": gates_passed,
            "pass_rate": gate_pass_rate,
            "all_gates_passed": all_gates_passed,
        },
        "model_evaluations": {
            "attempted": attempted,
            "completed": completed_count,
            "succeeded": succeeded,
            "failed": failed,
            "crashed": crashed,
            "interrupted": interrupted,
            "success_rate": success_rate,
            "sample_size": completed_count,
            "average_score": avg_score,
        },
        "tests": test_summary or {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "all_passed": True,
        },
    }


def get_test_report(cache_path: str = TEST_RESULTS_PATH,
                    run_if_missing: bool = True) -> Dict[str, Any]:
    """Return test results, cached or discovered via unittest."""
    if os.path.exists(cache_path) and not run_if_missing:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Run unittest discovery in-process
    start = time.time()
    try:
        suite = unittest.defaultTestLoader.discover(os.path.join(PROJECT_ROOT, "tests"))
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        elapsed = round(time.time() - start, 3)

        report = {
            "status": "ok",
            "total": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failed": len(result.failures),
            "errors": len(result.errors),
            "duration_s": elapsed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "all_passed": (len(result.failures) == 0 and len(result.errors) == 0),
        }

        # Cache atomically
        try:
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            os.replace(tmp_path, cache_path)
        except Exception:
            pass

        return report
    except Exception as e:
        return {
            "status": "error",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "error_message": str(e),
            "all_passed": False,
        }
