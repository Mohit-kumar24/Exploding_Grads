"""
KingdomForge Verifier

Independent programmatic verification of task outcomes.
ISOLATED from the agent -- the agent cannot read or modify this.

Checks actual kingdom state, not agent claims.
No LLM used. Pure programmatic checks.
"""
import time
import json
from typing import Dict, Any


class Verifier:
    """Programmatic verifier for KingdomForge tasks."""

    def __init__(self):
        self._max_verify_time = 120.0  # 2-minute limit

    def verify(self, env, task) -> Dict[str, Any]:
        """
        Verify task outcome by inspecting final kingdom state.

        Returns:
            score: float [0, 1]
            passed: bool
            details: verification breakdown
            classification: task_success, task_failure, integrity_violation, verifier_failure, harness_failure
            termination_reason, state_hash, steps_used, verification_time_ms
        """
        start_time = time.monotonic()
        classification = "task_failure"

        try:
            result = task.verify(env.state)
            score = float(result.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            details = result.get("details", {})
            passed = result.get("passed", score >= 0.95)
            classification = "task_success" if passed else "task_failure"
        except Exception as e:
            score = 0.0
            details = {"verification_error": str(e)}
            passed = False
            classification = "verifier_failure"

        # Check for state invariant and integrity violations
        if hasattr(env, "validate_invariants"):
            invariants = env.validate_invariants()
            if invariants:
                score = 0.0
                passed = False
                classification = "integrity_violation"
                details["invariant_violations"] = invariants

        if hasattr(env, "_discrepancies") and env._discrepancies:
            details["discrepancies"] = list(env._discrepancies)

        if env.termination_reason == "crash":
            classification = "harness_failure"

        elapsed_ms = (time.monotonic() - start_time) * 1000

        if elapsed_ms > self._max_verify_time * 1000:
            details["verification_timeout"] = True

        return {
            "score": round(score, 4),
            "passed": passed,
            "classification": classification,
            "details": details,
            "termination_reason": env.termination_reason or "unknown",
            "state_hash": env.state_hash(),
            "steps_used": env.step_count,
            "verification_time_ms": round(elapsed_ms, 2),
        }

    def verify_baseline(self, env, task, agent_type: str) -> Dict[str, Any]:
        """Verify a baseline run and check gate conditions."""
        result = self.verify(env, task)

        if agent_type == "reference":
            gate_passed = result["score"] >= 0.95
            result["gate_check"] = {
                "type": "reference",
                "threshold": 0.95,
                "actual": result["score"],
                "passed": gate_passed,
            }
        elif agent_type == "noop":
            gate_passed = result["score"] <= 0.05
            result["gate_check"] = {
                "type": "noop",
                "threshold": 0.05,
                "actual": result["score"],
                "passed": gate_passed,
            }
        else:
            gate_passed = False
            result["gate_check"] = {"error": "Unknown agent type: %s" % agent_type}

        return result


def create_verifier() -> Verifier:
    """Factory function to create a Verifier instance."""
    return Verifier()
