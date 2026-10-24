"""
Deterministic replay tests for KingdomForge.
Ensures:
1. Exact same seed + exact same actions produce identical state hash
2. Different actions produce different state hashes
3. Verifier produces identical scores on replayed states
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.database_env import KingdomEnv
from src.env.tasks import get_task
from src.agents.reference_agent import ReferenceAgent
from src.verifier.verifier import Verifier


class TestDeterministicReplay(unittest.TestCase):
    def test_deterministic_episode_replay(self):
        """Two runs with the same seed and actions must yield identical state hashes."""
        ref_agent = ReferenceAgent()
        verifier = Verifier()

        for tid in ["task_1", "task_2", "task_3"]:
            task = get_task(tid)

            # Run 1
            env1 = KingdomEnv()
            res1 = ref_agent.run_episode(env1, tid, seed=42)
            v1 = verifier.verify(env1, task)

            # Run 2
            env2 = KingdomEnv()
            res2 = ref_agent.run_episode(env2, tid, seed=42)
            v2 = verifier.verify(env2, task)

            self.assertEqual(
                env1.state_hash(), env2.state_hash(),
                f"State hashes differ for seed=42 on {tid}: {env1.state_hash()} != {env2.state_hash()}"
            )
            self.assertEqual(v1["score"], v2["score"])
            self.assertEqual(v1["passed"], v2["passed"])
            self.assertEqual(res1["steps"], res2["steps"])

    def test_divergent_actions_divergent_hash(self):
        """Different actions must lead to different state hashes."""
        env1 = KingdomEnv()
        env1.reset("task_1", seed=42)
        env1.step("build", {"building_type": "farm"})

        env2 = KingdomEnv()
        env2.reset("task_1", seed=42)
        env2.step("view_kingdom", {})

        self.assertNotEqual(env1.state_hash(), env2.state_hash())

    def test_step_by_step_full_trajectory_determinism(self):
        """
        Compare step-by-step across all 6 tasks:
        - initial state & hash
        - observation after every action
        - state hash after every action
        - final termination reason and verifier score
        """
        from src.env.tasks import get_all_task_ids
        ref_agent = ReferenceAgent()
        verifier = Verifier()

        for tid in get_all_task_ids():
            task = get_task(tid)

            # Replay 1
            env1 = KingdomEnv()
            obs1 = env1.reset(tid, seed=123)
            hash1_init = env1.state_hash()

            # Replay 2
            env2 = KingdomEnv()
            obs2 = env2.reset(tid, seed=123)
            hash2_init = env2.state_hash()

            self.assertEqual(hash1_init, hash2_init, f"Initial hashes diverge on {tid}")
            self.assertEqual(obs1, obs2, f"Initial observations diverge on {tid}")

            import random
            rng = random.Random(123)
            actions = task.get_reference_actions(rng)
            for step_idx, (tool, args) in enumerate(actions):
                res1, done1, info1 = env1.step(tool, args)
                res2, done2, info2 = env2.step(tool, args)

                self.assertEqual(res1, res2, f"Step {step_idx} observation diverges on {tid}")
                self.assertEqual(done1, done2, f"Step {step_idx} done flag diverges on {tid}")
                self.assertEqual(env1.state_hash(), env2.state_hash(), f"Step {step_idx} hash diverges on {tid}")

            v1 = verifier.verify(env1, task)
            v2 = verifier.verify(env2, task)
            self.assertEqual(v1["score"], v2["score"])
            self.assertEqual(v1["passed"], v2["passed"])
            self.assertEqual(env1.state_hash(), env2.state_hash())


if __name__ == "__main__":
    unittest.main()
