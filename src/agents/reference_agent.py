"""
Reference Agent for KingdomForge

Solves each task by executing pre-defined reference actions
through the same public tool interface available to any agent.
Must score >= 0.95 on every task.
"""
import os
import sys
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.env.tasks import get_task, get_all_task_ids
    from src.env.database_env import KingdomEnv
    from src.verifier.verifier import Verifier
except ImportError:
    from ..env.tasks import get_task, get_all_task_ids
    from ..env.database_env import KingdomEnv
    from ..verifier.verifier import Verifier


class ReferenceAgent:
    """Reference solution agent using known action sequences."""

    def __init__(self):
        self._action_index = 0
        self._actions: List[Tuple[str, Dict]] = []

    def reset(self, task_id: str, seed: int = 42):
        """Prepare reference solution for a specific task."""
        import random
        task = get_task(task_id)
        rng = random.Random(seed)
        self._actions = task.get_reference_actions(rng)
        self._action_index = 0

    def get_next_action(self) -> Tuple[str, Dict[str, Any]]:
        """Get the next action. Returns (None, None) if done."""
        if self._action_index >= len(self._actions):
            return None, None
        tool_name, args = self._actions[self._action_index]
        self._action_index += 1
        return tool_name, args

    def run_episode(self, env, task_id: str, seed: int = 42) -> Dict[str, Any]:
        """Run a complete episode using the reference solution."""
        obs = env.reset(task_id, seed)
        self.reset(task_id, seed)

        trajectory = []

        while True:
            tool_name, args = self.get_next_action()
            if tool_name is None:
                break

            result, done, info = env.step(tool_name, args)
            trajectory.append({
                "tool": tool_name,
                "args": args,
                "result": result[:500],
                "done": done,
                "state_hash": info.get("state_hash", ""),
            })

            if done:
                break

        return {
            "task_id": task_id,
            "seed": seed,
            "steps": len(trajectory),
            "trajectory": trajectory,
            "final_state_hash": env.state_hash(),
            "termination_reason": env.termination_reason,
        }


if __name__ == "__main__":
    task_id = sys.argv[1] if len(sys.argv) > 1 else "task_3"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    print(f"Running Reference Agent for {task_id} (seed={seed})...")
    agent = ReferenceAgent()
    env = KingdomEnv()
    res = agent.run_episode(env, task_id, seed)
    verifier = Verifier()
    task = get_task(task_id)
    v = verifier.verify(env, task)
    print(f"Verification: score={v['score']:.4f}, passed={v['passed']}")
    sys.exit(0 if v["passed"] else 1)
