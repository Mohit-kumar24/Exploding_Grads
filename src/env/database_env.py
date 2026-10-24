"""
KingdomForge Environment

A medieval kingdom management simulation. The agent manages resources,
builds structures, assigns workers, trades at the market, and must
achieve specific objectives within turn and step limits.

Core mechanics:
- Resources: gold, food, wood, stone
- Population: total people, idle workers, assigned workers
- Buildings: each type has costs, capacity, and production rates
- Turns: advancing a turn triggers production, consumption, and events
- Seeded RNG: same seed = same starting state and event sequence
"""
import hashlib
import json
import random
import copy
import time
from typing import Any, Dict, List, Optional, Tuple


# ---- Building definitions ----
BUILDING_DEFS = {
    "farm":        {"cost": {"wood": 10, "gold": 5},   "worker_cap": 3, "production": {"food": 4},   "description": "Produces food when workers are assigned"},
    "lumber_mill": {"cost": {"wood": 5,  "gold": 10},  "worker_cap": 3, "production": {"wood": 3},   "description": "Produces wood when workers are assigned"},
    "quarry":      {"cost": {"wood": 15, "gold": 15},  "worker_cap": 3, "production": {"stone": 2},  "description": "Produces stone when workers are assigned"},
    "house":       {"cost": {"wood": 15, "gold": 10},  "worker_cap": 0, "production": {},            "description": "Increases population capacity by 5", "pop_cap": 5},
    "barracks":    {"cost": {"wood": 20, "stone": 10, "gold": 20}, "worker_cap": 0, "production": {},  "description": "Military building, required for defense tasks"},
    "market":      {"cost": {"wood": 20, "gold": 15},  "worker_cap": 2, "production": {"gold": 5},   "description": "Enables trading and produces gold"},
    "granary":     {"cost": {"wood": 15, "stone": 5},   "worker_cap": 0, "production": {},            "description": "Increases food storage capacity by 50"},
    "wall":        {"cost": {"stone": 20, "gold": 10},  "worker_cap": 0, "production": {},            "description": "Defensive structure, adds +10 defense"},
    "workshop":    {"cost": {"wood": 15, "stone": 10, "gold": 20}, "worker_cap": 2, "production": {},  "description": "Enables advanced building upgrades"},
    "castle":      {"cost": {"wood": 50, "stone": 50, "gold": 100}, "worker_cap": 0, "production": {},  "description": "The ultimate structure. Requires workshop."},
}

# Trade rates at the market
TRADE_RATES = {
    "sell": {"food": 2, "wood": 3, "stone": 4},     # resource -> gold
    "buy":  {"food": 3, "wood": 5, "stone": 7},     # gold -> resource
}


class KingdomEnv:
    """
    Kingdom management simulation environment.

    The agent interacts through tool calls. State is fully deterministic
    given the same seed and action sequence.
    """

    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._seed: Optional[int] = None
        self._rng: Optional[random.Random] = None
        self._step_count: int = 0
        self._turn: int = 0
        self._done: bool = False
        self._terminated: bool = False
        self._termination_reason: Optional[str] = None
        self._submitted_answer: Optional[str] = None
        self._task_id: Optional[str] = None
        self._task = None
        self._action_log: List[Dict] = []
        self._start_time: Optional[float] = None
        self._max_steps: int = 40
        self._max_time: float = 600.0  # 10 minutes
        self._event_log: List[str] = []
        self._discrepancies: List[Dict] = []

    def validate_invariants(self) -> List[str]:
        """Validate explicit kingdom state invariants."""
        violations = []
        res = self._state.get("resources", {})
        for r, amt in res.items():
            if amt < 0:
                violations.append("Negative resource: %s=%d" % (r, amt))

        pop = self._state.get("population", {})
        total_pop = pop.get("total", 0)
        idle_pop = pop.get("idle", 0)
        total_assigned = sum(self._state.get("workers", {}).values())
        if total_pop < 0:
            violations.append("Negative total population: %d" % total_pop)
        if idle_pop < 0:
            violations.append("Negative idle population: %d" % idle_pop)
        if total_assigned + idle_pop != total_pop:
            violations.append("Population mismatch: assigned(%d) + idle(%d) != total(%d)" % (
                total_assigned, idle_pop, total_pop))

        for btype, assigned in self._state.get("workers", {}).items():
            if assigned < 0:
                violations.append("Negative workers assigned to %s: %d" % (btype, assigned))
            bcount = self._state.get("buildings", {}).get(btype, 0)
            bcap = BUILDING_DEFS.get(btype, {}).get("worker_cap", 0)
            max_allowed = bcount * bcap
            if assigned > max_allowed:
                violations.append("Worker capacity exceeded at %s: %d > %d" % (btype, assigned, max_allowed))

        if self._state.get("defense", 0) < 0:
            violations.append("Negative defense: %d" % self._state.get("defense", 0))

        return violations

    def _format_error(self, tool_name: str, reason: str, constraint: str, recovery: str) -> str:
        """Structured error feedback compliant with live-run diagnostics."""
        return (
            "Error: %s\n"
            "INVALID_ACTION: %s\n"
            "Reason: %s\n"
            "Constraint: %s\n"
            "State change: none. The action was rejected before mutation.\n"
            "Recovery: %s"
        ) % (reason, tool_name, reason, constraint, recovery)

    def reset(self, task_id: str, seed: int = 42) -> Dict[str, Any]:
        """
        Reset the environment for a specific task with a given seed.
        Returns the initial observation.
        """
        from .tasks import get_task

        self._seed = seed
        self._rng = random.Random(seed)
        self._step_count = 0
        self._turn = 1
        self._done = False
        self._terminated = False
        self._termination_reason = None
        self._submitted_answer = None
        self._task_id = task_id
        self._action_log = []
        self._start_time = time.monotonic()
        self._event_log = []
        self._discrepancies = []

        # Default starting state
        self._state = {
            "resources": {"gold": 50, "food": 30, "wood": 30, "stone": 10},
            "population": {"total": 5, "idle": 5, "capacity": 10},
            "buildings": {},       # type -> count
            "workers": {},         # building_type -> assigned_count
            "defense": 0,
            "food_storage_cap": 100,
            "turn": 1,
            "max_turns": 30,
            "events": [],
            "submitted_answer": "",
            "failed_actions_count": 0,
        }

        # Load and setup the task
        self._task = get_task(task_id)
        self._task.setup(self._state, self._rng)

        observation = {
            "task_id": task_id,
            "task_instruction": self._task.instruction,
            "difficulty": self._task.difficulty,
            "initial_state_hash": self.state_hash(),
            "kingdom": self._get_visible_state(),
            "seed": seed,
        }
        return observation

    def set_max_steps(self, max_steps: int):
        """Set the maximum number of tool call actions allowed before termination."""
        self._max_steps = max(1, int(max_steps))

    def step(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool, Dict]:
        """
        Execute a tool call. Returns (observation_str, done, info).
        """
        # Block post-termination actions
        if self._done:
            return (
                "Error: Episode has already ended. No further actions are accepted.",
                True,
                {"error": "post_termination_action", "state_hash": self.state_hash()}
            )

        # Check step limit: reject if already at or beyond max allowed actions
        if self._step_count >= self._max_steps:
            self._done = True
            self._terminated = True
            self._termination_reason = "step_cap"
            return (
                "Error: Maximum steps (%d) exceeded. Episode terminated." % self._max_steps,
                True,
                {"termination_reason": "step_cap", "state_hash": self.state_hash()}
            )

        self._step_count += 1
        pre_hash = self.state_hash()

        # Check time limit
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        if elapsed > self._max_time:
            self._done = True
            self._terminated = True
            self._termination_reason = "time_cap"
            return (
                "Error: Time limit exceeded. Episode terminated.",
                True,
                {"termination_reason": "time_cap", "state_hash": self.state_hash()}
            )

        # Validate tool name
        from .tools import VALID_TOOL_NAMES
        if tool_name not in VALID_TOOL_NAMES:
            info = {
                "error": "invalid_tool",
                "tool_name": tool_name,
                "valid_tools": sorted(VALID_TOOL_NAMES),
                "state_hash": self.state_hash(),
                "step": self._step_count
            }
            return (
                "Error: Unknown tool '%s'. Valid tools: %s" % (tool_name, sorted(VALID_TOOL_NAMES)),
                False,
                info
            )

        # Execute the tool
        try:
            result = self._execute_tool(tool_name, args)
        except Exception as e:
            result = "Error executing %s: %s" % (tool_name, str(e))

        post_hash = self.state_hash()
        is_error = isinstance(result, str) and result.startswith("Error")
        discrepancies = []

        # Discrepancy check 1: Error returned but state mutated
        if is_error and pre_hash != post_hash:
            disc = {
                "category": "error_state_mutation",
                "severity": "high",
                "tool": tool_name,
                "step": self._step_count,
                "message": "Tool '%s' reported error but state hash changed from %s to %s" % (tool_name, pre_hash, post_hash)
            }
            discrepancies.append(disc)
            self._discrepancies.append(disc)

        # Discrepancy check 2: State invariant violations
        inv_violations = self.validate_invariants()
        if inv_violations:
            disc = {
                "category": "state_invariant_violation",
                "severity": "critical",
                "tool": tool_name,
                "step": self._step_count,
                "violations": inv_violations,
                "message": "Invariants violated after '%s': %s" % (tool_name, ", ".join(inv_violations))
            }
            discrepancies.append(disc)
            self._discrepancies.append(disc)

        if self._step_count >= self._max_steps:
            self._done = True
            self._terminated = True
            if not self._termination_reason:
                self._termination_reason = "step_cap"

        done = self._done
        info = {
            "step": self._step_count,
            "state_hash": post_hash,
            "pre_state_hash": pre_hash,
            "tool_name": tool_name,
            "turn": self._state.get("turn", self._turn),
            "discrepancies": discrepancies,
        }
        if self._termination_reason:
            info["termination_reason"] = self._termination_reason

        # Log the action
        self._action_log.append({
            "step": self._step_count,
            "tool_name": tool_name,
            "args": args,
            "result_preview": str(result)[:500],
            "pre_state_hash": pre_hash,
            "state_hash": post_hash,
            "discrepancies": discrepancies,
            "timestamp": time.time()
        })

        # Truncate result for context management
        if isinstance(result, str) and len(result) > 4000:
            result = result[:4000] + "\n... (output truncated to 4000 chars)"

        return (str(result), done, info)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a validated tool call."""
        if tool_name == "view_kingdom":
            return self._tool_view_kingdom()
        elif tool_name == "build":
            return self._tool_build(args)
        elif tool_name == "assign_workers":
            return self._tool_assign_workers(args)
        elif tool_name == "unassign_workers":
            return self._tool_unassign_workers(args)
        elif tool_name == "trade":
            return self._tool_trade(args)
        elif tool_name == "recruit":
            return self._tool_recruit(args)
        elif tool_name == "advance_turn":
            return self._tool_advance_turn()
        elif tool_name == "check_objectives":
            return self._tool_check_objectives()
        elif tool_name == "submit_answer":
            return self._tool_submit_answer(args)
        else:
            return "Error: Tool '%s' not implemented" % tool_name

    # ---- Tool implementations ----

    def _tool_view_kingdom(self) -> str:
        return json.dumps(self._get_visible_state(), indent=2)

    def _tool_build(self, args: Dict) -> str:
        btype = args.get("building_type", "")
        if btype not in BUILDING_DEFS:
            return self._format_error(
                "build",
                "Unknown building '%s'. Available: %s" % (btype, list(BUILDING_DEFS.keys())),
                "Must be a recognized building type from BUILDING_DEFS",
                "Inspect the kingdom building catalog and choose a valid building."
            )

        bdef = BUILDING_DEFS[btype]

        # Check prerequisite: castle needs workshop
        if btype == "castle":
            if self._state["buildings"].get("workshop", 0) < 1:
                return self._format_error(
                    "build",
                    "Building a castle requires at least one workshop.",
                    "Prerequisite: completed workshop >= 1",
                    "Construct a workshop first before attempting to build a castle."
                )

        # Check resource costs
        for res, cost in bdef["cost"].items():
            have = self._state["resources"].get(res, 0)
            if have < cost:
                return self._format_error(
                    "build",
                    "Not enough %s. Need %d, have %d." % (res, cost, have),
                    "Resource requirement: %d %s" % (cost, res),
                    "Produce or trade for %d more %s before constructing %s." % (cost - have, res, btype)
                )

        # Deduct costs
        for res, cost in bdef["cost"].items():
            self._state["resources"][res] -= cost

        # Add building
        self._state["buildings"][btype] = self._state["buildings"].get(btype, 0) + 1

        # Apply building effects
        effects = []
        if "pop_cap" in bdef:
            self._state["population"]["capacity"] += bdef["pop_cap"]
            effects.append("Population capacity +%d" % bdef["pop_cap"])

        if btype == "granary":
            self._state["food_storage_cap"] += 50
            effects.append("Food storage capacity +50")

        if btype == "wall":
            self._state["defense"] += 10
            effects.append("Defense +10")

        if btype == "barracks":
            self._state["defense"] += 5
            effects.append("Defense +5")

        effect_str = "; ".join(effects) if effects else "No immediate effects"
        return "Built %s successfully. %s. Buildings: %s" % (
            btype, effect_str, json.dumps(self._state["buildings"]))

    def _tool_assign_workers(self, args: Dict) -> str:
        btype = args.get("building_type", "")
        count = args.get("count", 0)
        try:
            if isinstance(count, str) and count.strip().isdigit():
                count = int(count.strip())
        except Exception:
            pass

        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            return self._format_error(
                "assign_workers",
                "count must be a positive integer.",
                "count must be an integer >= 1",
                "Specify a positive integer count of workers to assign."
            )

        if btype not in BUILDING_DEFS:
            return self._format_error(
                "assign_workers",
                "Unknown building '%s'." % btype,
                "Building type must exist in catalog",
                "Specify a valid building type from the kingdom catalog."
            )

        bdef = BUILDING_DEFS[btype]
        if bdef["worker_cap"] == 0:
            return self._format_error(
                "assign_workers",
                "Building '%s' does not accept workers." % btype,
                "Worker capacity for '%s' is 0" % btype,
                "Assign workers to production structures (farm, lumber_mill, quarry, market)."
            )

        # Check building exists
        if self._state["buildings"].get(btype, 0) < 1:
            return self._format_error(
                "assign_workers",
                "No '%s' building exists. Build one first." % btype,
                "Must own at least 1 structure before assigning workers",
                "Construct a %s before assigning workers to it." % btype
            )

        # Check idle workers
        idle = self._state["population"]["idle"]
        if idle < count:
            return self._format_error(
                "assign_workers",
                "Not enough idle workers. Idle: %d, requested: %d." % (idle, count),
                "Assigned count cannot exceed idle population pool",
                "Unassign workers from other buildings or recruit new workers first."
            )

        # Check worker capacity
        max_workers = self._state["buildings"][btype] * bdef["worker_cap"]
        current_workers = self._state["workers"].get(btype, 0)
        available_slots = max_workers - current_workers
        if count > available_slots:
            return self._format_error(
                "assign_workers",
                "Not enough worker slots in '%s'. Available slots: %d." % (btype, available_slots),
                "Cannot exceed total building worker capacity (%d slots)" % max_workers,
                "Construct additional %s structures to open more worker slots." % btype
            )

        # Assign
        self._state["workers"][btype] = current_workers + count
        self._state["population"]["idle"] -= count

        return "Assigned %d worker(s) to %s. Workers at %s: %d/%d. Idle workers: %d." % (
            count, btype, btype, self._state["workers"][btype], max_workers,
            self._state["population"]["idle"])

    def _tool_unassign_workers(self, args: Dict) -> str:
        btype = args.get("building_type", "")
        count = args.get("count", 0)
        try:
            if isinstance(count, str) and count.strip().isdigit():
                count = int(count.strip())
        except Exception:
            pass

        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            return self._format_error(
                "unassign_workers",
                "count must be a positive integer.",
                "count must be an integer >= 1",
                "Specify a positive integer count of workers to unassign."
            )

        current = self._state["workers"].get(btype, 0)
        if count > current:
            return self._format_error(
                "unassign_workers",
                "Only %d workers assigned to '%s'." % (current, btype),
                "Cannot unassign more workers than currently assigned",
                "Unassign at most %d workers from %s." % (current, btype)
            )

        self._state["workers"][btype] = current - count
        self._state["population"]["idle"] += count

        return "Unassigned %d worker(s) from %s. Idle workers: %d." % (
            count, btype, self._state["population"]["idle"])

    def _tool_trade(self, args: Dict) -> str:
        action = args.get("action", "")
        resource = args.get("resource", "")
        amount = args.get("amount", 0)
        try:
            if isinstance(amount, str) and amount.strip().isdigit():
                amount = int(amount.strip())
        except Exception:
            pass

        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
            return self._format_error(
                "trade",
                "amount must be a positive integer.",
                "amount must be an integer >= 1",
                "Specify a positive integer trade amount."
            )

        if self._state["buildings"].get("market", 0) < 1:
            return self._format_error(
                "trade",
                "You need a market to trade. Build one first.",
                "Market structure required for trade operations",
                "Construct a market before attempting trade actions."
            )

        if resource not in ("food", "wood", "stone"):
            return self._format_error(
                "trade",
                "Can only trade food, wood, or stone.",
                "Permitted trade commodities: food, wood, stone",
                "Select a valid commodity (food, wood, or stone)."
            )

        if action == "sell":
            # Sell resource for gold
            have = self._state["resources"].get(resource, 0)
            if have < amount:
                return self._format_error(
                    "trade",
                    "Not enough %s. Have: %d, want to sell: %d." % (resource, have, amount),
                    "Cannot sell more resources than held in storage",
                    "Sell at most %d %s or produce more before selling." % (have, resource)
                )
            rate = TRADE_RATES["sell"][resource]
            gold_gained = amount * rate
            self._state["resources"][resource] -= amount
            self._state["resources"]["gold"] += gold_gained
            return "Sold %d %s for %d gold. Gold: %d, %s: %d." % (
                amount, resource, gold_gained,
                self._state["resources"]["gold"],
                resource, self._state["resources"][resource])

        elif action == "buy":
            # Buy resource with gold
            rate = TRADE_RATES["buy"][resource]
            total_cost = amount * rate
            have_gold = self._state["resources"]["gold"]
            if have_gold < total_cost:
                return self._format_error(
                    "trade",
                    "Not enough gold. Need: %d, have: %d." % (total_cost, have_gold),
                    "Treasury gold must cover purchase cost (%d gold)" % total_cost,
                    "Sell surplus resources or reduce purchase amount to afford trade."
                )
            self._state["resources"]["gold"] -= total_cost
            self._state["resources"][resource] += amount
            return "Bought %d %s for %d gold. Gold: %d, %s: %d." % (
                amount, resource, total_cost,
                self._state["resources"]["gold"],
                resource, self._state["resources"][resource])
        else:
            return self._format_error(
                "trade",
                "action must be 'buy' or 'sell'.",
                "Trade action must be either 'buy' or 'sell'",
                "Specify action='buy' or action='sell'."
            )

    def _tool_recruit(self, args: Dict) -> str:
        count = args.get("count", 0)
        try:
            if isinstance(count, str) and count.strip().isdigit():
                count = int(count.strip())
        except Exception:
            pass

        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            return self._format_error(
                "recruit",
                "count must be a positive integer.",
                "count must be an integer >= 1",
                "Specify a positive integer count of recruits."
            )

        # Check capacity
        pop = self._state["population"]
        available_cap = pop["capacity"] - pop["total"]
        if count > available_cap:
            return self._format_error(
                "recruit",
                "Not enough population capacity. Available: %d. Build more houses." % available_cap,
                "Total population cannot exceed capacity (%d)" % pop["capacity"],
                "Construct housing (+5 capacity each) before recruiting more workers."
            )

        # Cost: 10 gold + 5 food per worker
        gold_cost = 10 * count
        food_cost = 5 * count
        have_gold = self._state["resources"]["gold"]
        have_food = self._state["resources"]["food"]
        if have_gold < gold_cost:
            return self._format_error(
                "recruit",
                "Not enough gold. Need %d, have %d." % (gold_cost, have_gold),
                "Recruitment cost: 10 gold per worker",
                "Accumulate at least %d gold before recruiting %d workers." % (gold_cost, count)
            )
        if have_food < food_cost:
            return self._format_error(
                "recruit",
                "Not enough food. Need %d, have %d." % (food_cost, have_food),
                "Recruitment cost: 5 food per worker",
                "Store at least %d food before recruiting %d workers." % (food_cost, count)
            )

        self._state["resources"]["gold"] -= gold_cost
        self._state["resources"]["food"] -= food_cost
        self._state["population"]["total"] += count
        self._state["population"]["idle"] += count

        return "Recruited %d worker(s). Total pop: %d/%d. Idle: %d. Gold: %d, Food: %d." % (
            count, pop["total"], pop["capacity"], pop["idle"],
            self._state["resources"]["gold"], self._state["resources"]["food"])

    def _tool_advance_turn(self) -> str:
        turn = self._state.get("turn", self._turn)
        max_turns = self._state.get("max_turns", 30)

        if turn >= max_turns:
            self._done = True
            self._termination_reason = "turn_limit"
            return "Turn limit reached (%d/%d). Episode ended." % (turn, max_turns)

        # Production phase
        production_report = []
        for btype, worker_count in self._state["workers"].items():
            if worker_count > 0 and btype in BUILDING_DEFS:
                bdef = BUILDING_DEFS[btype]
                for res, rate in bdef["production"].items():
                    produced = rate * worker_count
                    self._state["resources"][res] += produced
                    production_report.append("+%d %s from %s" % (produced, res, btype))

        # Food consumption: 1 food per person per turn
        food_consumed = self._state["population"]["total"]
        self._state["resources"]["food"] -= food_consumed
        production_report.append("-%d food consumed by population" % food_consumed)

        # Cap food storage
        if self._state["resources"]["food"] > self._state["food_storage_cap"]:
            self._state["resources"]["food"] = self._state["food_storage_cap"]

        # Starvation check
        starvation_msg = ""
        if self._state["resources"]["food"] < 0:
            food_deficit = abs(self._state["resources"]["food"])
            people_lost = min(food_deficit, self._state["population"]["total"])

            # 1. Deduct first from idle pool
            idle_lost = min(people_lost, self._state["population"]["idle"])
            self._state["population"]["idle"] -= idle_lost
            remaining_lost = people_lost - idle_lost

            # 2. If idle exhausted, pull starving population from assigned workers
            if remaining_lost > 0:
                for btype in list(self._state["workers"].keys()):
                    assigned = self._state["workers"][btype]
                    if assigned > 0:
                        to_remove = min(remaining_lost, assigned)
                        self._state["workers"][btype] -= to_remove
                        remaining_lost -= to_remove
                        if remaining_lost == 0:
                            break

            self._state["population"]["total"] -= people_lost
            self._state["resources"]["food"] = 0
            if people_lost > 0:
                starvation_msg = " STARVATION: %d people left the kingdom!" % people_lost
                self._event_log.append("turn_%d_starvation_%d_lost" % (turn + 1, people_lost))

        # Random event (deterministic from seed)
        event_msg = ""
        event_roll = self._rng.random()
        if event_roll < 0.15 and turn > 3:
            # Bandit raid
            defense = self._state.get("defense", 0)
            if defense < 10:
                gold_lost = min(self._rng.randint(5, 15), self._state["resources"]["gold"])
                self._state["resources"]["gold"] -= gold_lost
                event_msg = " EVENT: Bandit raid! Lost %d gold (low defense)." % gold_lost
                self._event_log.append("turn_%d_bandit_raid_lost_%d_gold" % (turn + 1, gold_lost))
            else:
                event_msg = " EVENT: Bandits spotted but your defenses held!"
                self._event_log.append("turn_%d_bandits_repelled" % (turn + 1))
        elif event_roll < 0.25 and turn > 2:
            # Good harvest
            bonus_food = self._rng.randint(3, 8)
            self._state["resources"]["food"] += bonus_food
            event_msg = " EVENT: Bountiful harvest! +%d food." % bonus_food
            self._event_log.append("turn_%d_good_harvest_%d" % (turn + 1, bonus_food))

        # Advance turn counter
        self._state["turn"] = turn + 1
        self._turn = turn + 1
        self._state["events"] = list(self._event_log)

        report = "Turn %d -> %d. %s%s%s Resources: %s" % (
            turn, turn + 1,
            " ".join(production_report) if production_report else "No production.",
            starvation_msg, event_msg,
            json.dumps(self._state["resources"]))

        return report

    def _tool_check_objectives(self) -> str:
        if self._task is None:
            return "Error: No task loaded."
        progress = self._task.check_progress(self._state)
        return json.dumps(progress, indent=2)

    def _tool_submit_answer(self, args: Dict) -> str:
        answer = args.get("answer", "")
        self._submitted_answer = str(answer)
        self._state["submitted_answer"] = str(answer)
        self._done = True
        self._termination_reason = "submitted"
        return "Answer submitted: %s" % str(answer)[:200]

    # ---- Internal methods ----

    def _get_visible_state(self) -> Dict:
        """Return the state visible to the agent (no hidden internals)."""
        return {
            "resources": dict(self._state["resources"]),
            "population": dict(self._state["population"]),
            "buildings": dict(self._state["buildings"]),
            "workers": dict(self._state["workers"]),
            "defense": self._state["defense"],
            "food_storage_cap": self._state["food_storage_cap"],
            "turn": self._state.get("turn", self._turn),
            "max_turns": self._state.get("max_turns", 30),
            "recent_events": list(self._event_log[-5:]) if self._event_log else [],
        }

    def state_hash(self) -> str:
        """
        Deterministic hash of entire state. Uses stable serialization.
        """
        hashable = {
            "resources": dict(sorted(self._state["resources"].items())),
            "population": dict(sorted(self._state["population"].items())),
            "buildings": dict(sorted(self._state["buildings"].items())),
            "workers": dict(sorted(self._state["workers"].items())),
            "defense": self._state["defense"],
            "food_storage_cap": self._state["food_storage_cap"],
            "turn": self._state.get("turn", self._turn),
            "events": sorted(self._event_log),
        }
        state_str = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()[:8]

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def done(self) -> bool:
        return self._done

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def termination_reason(self) -> Optional[str]:
        return self._termination_reason

    @property
    def state(self) -> Dict:
        """Direct state access for verifier (NOT exposed to agent)."""
        return self._state

    @property
    def discrepancies(self) -> List[Dict]:
        """Detected discrepancies in environment state or execution."""
        return list(self._discrepancies)

    @property
    def turn(self) -> int:
        return self._state.get("turn", self._turn)

    def close(self):
        """Cleanup."""
        pass
