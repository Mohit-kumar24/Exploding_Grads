"""
No-Op Agent for KingdomForge

Gives up immediately without doing anything useful.
Must score <= 0.05 on every task.
Validates that the verifier does not award undeserved reward.
"""
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.env.tasks import get_task
    from src.env.database_env import KingdomEnv
    from src.verifier.verifier import Verifier
except ImportError:
    from ..env.tasks import get_task
    from ..env.database_env import KingdomEnv
    from ..verifier.verifier import Verifier


class NoopAgent:
    """No-op agent that immediately submits without useful actions."""

    def run_episode(self, env, task_id: str, seed: int = 42) -> Dict[str, Any]:
        """Run a no-op episode: reset and immediately submit."""
        obs = env.reset(task_id, seed)

        result, done, info = env.step("submit_answer", {"answer": ""})

        return {
            "task_id": task_id,
            "seed": seed,
            "steps": 1,
            "trajectory": [{
                "tool": "submit_answer",
                "args": {"answer": ""},
                "result": result[:200],
                "done": done,
                "state_hash": info.get("state_hash", ""),
            }],
            "final_state_hash": env.state_hash(),
            "termination_reason": env.termination_reason,
        }


if __name__ == "__main__":
    task_id = sys.argv[1] if len(sys.argv) > 1 else "task_3"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    print(f"Running No-Op Agent for {task_id} (seed={seed})...")
    agent = NoopAgent()
    env = KingdomEnv()
    res = agent.run_episode(env, task_id, seed)
    verifier = Verifier()
    task = get_task(task_id)
    v = verifier.verify(env, task)
    gate_passed = v["score"] <= 0.05
    print(f"Verification: score={v['score']:.4f}, gate_passed={gate_passed}")
    sys.exit(0 if gate_passed else 1)
