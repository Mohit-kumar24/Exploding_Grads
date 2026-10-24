"""
KingdomForge Task Definitions

6 tasks across 3 difficulty levels (easy, medium, hard).
Each task defines setup, objectives, verification, and reference actions.

Reward Formula:
    score = sum(objective_i_weight * objective_i_met) for all objectives
    Each objective is binary (0 or 1) based on programmatic state checks.
    Partial credit given for partial objective completion where applicable.
    score is clamped to [0.0, 1.0].
"""
import json


class Task:
    """A single task definition."""

    def __init__(self, task_id, difficulty, title, instruction, description,
                 setup_fn, verify_fn, check_progress_fn, reference_actions_fn):
        self.task_id = task_id
        self.difficulty = difficulty
        self.title = title
        self.instruction = instruction
        self.description = description
        self._setup_fn = setup_fn
        self._verify_fn = verify_fn
        self._check_progress_fn = check_progress_fn
        self._reference_actions_fn = reference_actions_fn

    def setup(self, state, rng):
        """Initialize the kingdom state for this task."""
        self._setup_fn(state, rng)

    def verify(self, state):
        """
        Verify task outcome. Returns dict with:
        - score: float [0, 1]
        - details: verification breakdown
        - passed: bool
        """
        return self._verify_fn(state)

    def check_progress(self, state):
        """Return current progress toward objectives."""
        return self._check_progress_fn(state)

    def get_reference_actions(self, rng):
        """Return list of (tool_name, args_dict) for reference solution."""
        return self._reference_actions_fn(rng)


# ============================================================
# Task 1: EASY -- Build a Farm and Produce Food
# ============================================================

def _setup_task1(state, rng):
    """Start with basics. Agent must build a farm and produce food."""
    state["resources"] = {"gold": 30, "food": 20, "wood": 20, "stone": 5}
    state["population"] = {"total": 5, "idle": 5, "capacity": 10}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 15

def _verify_task1(state):
    details = {}
    score = 0.0

    # Objective 1: Build at least 1 farm (weight: 0.3)
    farm_count = state["buildings"].get("farm", 0)
    if farm_count >= 1:
        details["farm_built"] = True
        score += 0.3
    else:
        details["farm_built"] = False

    # Objective 2: Have at least 2 workers assigned to farms (weight: 0.3)
    farm_workers = state["workers"].get("farm", 0)
    if farm_workers >= 2:
        details["workers_assigned"] = True
        score += 0.3
    else:
        details["workers_assigned"] = False
        details["farm_workers"] = farm_workers

    # Objective 3: Have at least 40 food (weight: 0.4)
    food = state["resources"].get("food", 0)
    if food >= 40:
        details["food_target_met"] = True
        score += 0.4
    else:
        details["food_target_met"] = False
        details["food_current"] = food
        # Partial credit
        if food >= 25:
            score += 0.2

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task1(state):
    return {
        "objectives": [
            {"name": "Build at least 1 farm", "met": state["buildings"].get("farm", 0) >= 1,
             "current": state["buildings"].get("farm", 0), "target": 1},
            {"name": "Assign at least 2 workers to farms", "met": state["workers"].get("farm", 0) >= 2,
             "current": state["workers"].get("farm", 0), "target": 2},
            {"name": "Accumulate at least 40 food", "met": state["resources"].get("food", 0) >= 40,
             "current": state["resources"].get("food", 0), "target": 40},
        ]
    }

def _ref_actions_task1(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("assign_workers", {"building_type": "farm", "count": 3}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 2: EASY -- Build Infrastructure
# ============================================================

def _setup_task2(state, rng):
    """Build basic infrastructure: farm, lumber mill, and house."""
    state["resources"] = {"gold": 50, "food": 25, "wood": 40, "stone": 10}
    state["population"] = {"total": 5, "idle": 5, "capacity": 5}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 20

def _verify_task2(state):
    details = {}
    score = 0.0

    # Objective 1: Build farm, lumber_mill, house (each 0.2)
    for btype in ["farm", "lumber_mill", "house"]:
        if state["buildings"].get(btype, 0) >= 1:
            details["%s_built" % btype] = True
            score += 0.2
        else:
            details["%s_built" % btype] = False

    # Objective 2: Population capacity >= 10 (weight: 0.15)
    if state["population"]["capacity"] >= 10:
        details["pop_capacity_met"] = True
        score += 0.15
    else:
        details["pop_capacity_met"] = False

    # Objective 3: At least 6 total population (weight: 0.25)
    if state["population"]["total"] >= 6:
        details["pop_total_met"] = True
        score += 0.25
    else:
        details["pop_total_met"] = False
        details["pop_total"] = state["population"]["total"]

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task2(state):
    return {
        "objectives": [
            {"name": "Build a farm", "met": state["buildings"].get("farm", 0) >= 1},
            {"name": "Build a lumber mill", "met": state["buildings"].get("lumber_mill", 0) >= 1},
            {"name": "Build a house", "met": state["buildings"].get("house", 0) >= 1},
            {"name": "Population capacity >= 10", "met": state["population"]["capacity"] >= 10,
             "current": state["population"]["capacity"]},
            {"name": "Total population >= 6", "met": state["population"]["total"] >= 6,
             "current": state["population"]["total"]},
        ]
    }

def _ref_actions_task2(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "lumber_mill"}),
        ("build", {"building_type": "house"}),
        ("assign_workers", {"building_type": "farm", "count": 2}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("recruit", {"count": 1}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 3: MEDIUM -- Economy Building (DEMO TASK)
# ============================================================

def _setup_task3(state, rng):
    """Build a functioning economy with market and trade."""
    state["resources"] = {"gold": 40, "food": 25, "wood": 30, "stone": 10}
    state["population"] = {"total": 5, "idle": 5, "capacity": 10}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 25

def _verify_task3(state):
    details = {}
    score = 0.0

    # Objective 1: Build a market (0.20)
    if state["buildings"].get("market", 0) >= 1:
        details["market_built"] = True
        score += 0.20
    else:
        details["market_built"] = False

    # Objective 2: Build at least 2 farms (0.20)
    if state["buildings"].get("farm", 0) >= 2:
        details["farms_built"] = True
        score += 0.20
    else:
        details["farms_built"] = False
        details["farm_count"] = state["buildings"].get("farm", 0)

    # Objective 3: Have at least 60 gold (0.20)
    gold = state["resources"].get("gold", 0)
    if gold >= 60:
        details["gold_target_met"] = True
        score += 0.20
    else:
        details["gold_target_met"] = False
        details["gold_current"] = gold

    # Objective 4: Have at least 7 total population (0.20)
    pop = state["population"]["total"]
    if pop >= 7:
        details["pop_target_met"] = True
        score += 0.20
    else:
        details["pop_target_met"] = False
        details["pop_current"] = pop

    # Objective 5: Have at least 35 food (0.20)
    food = state["resources"].get("food", 0)
    if food >= 35:
        details["food_target_met"] = True
        score += 0.20
    else:
        details["food_target_met"] = False
        details["food_current"] = food

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task3(state):
    return {
        "objectives": [
            {"name": "Build a market", "met": state["buildings"].get("market", 0) >= 1},
            {"name": "Build at least 2 farms", "met": state["buildings"].get("farm", 0) >= 2,
             "current": state["buildings"].get("farm", 0)},
            {"name": "Accumulate 60+ gold", "met": state["resources"].get("gold", 0) >= 60,
             "current": state["resources"].get("gold", 0)},
            {"name": "Grow population to 7+", "met": state["population"]["total"] >= 7,
             "current": state["population"]["total"]},
            {"name": "Accumulate 35+ food", "met": state["resources"].get("food", 0) >= 35,
             "current": state["resources"].get("food", 0)},
        ]
    }

def _ref_actions_task3(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "lumber_mill"}),
        ("assign_workers", {"building_type": "farm", "count": 2}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("build", {"building_type": "farm"}),
        ("advance_turn", {}),
        ("build", {"building_type": "market"}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("trade", {"action": "sell", "resource": "food", "amount": 15}),
        ("advance_turn", {}),
        ("trade", {"action": "sell", "resource": "wood", "amount": 15}),
        ("recruit", {"count": 2}),
        ("assign_workers", {"building_type": "farm", "count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("trade", {"action": "sell", "resource": "food", "amount": 15}),
        ("trade", {"action": "sell", "resource": "wood", "amount": 10}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 4: MEDIUM -- Defend the Kingdom
# ============================================================

def _setup_task4(state, rng):
    """Must build defenses before bandits become more aggressive."""
    state["resources"] = {"gold": 80, "food": 30, "wood": 50, "stone": 30}
    state["population"] = {"total": 6, "idle": 6, "capacity": 10}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 25

def _verify_task4(state):
    details = {}
    score = 0.0

    # Objective 1: Build at least 1 barracks (0.25)
    if state["buildings"].get("barracks", 0) >= 1:
        details["barracks_built"] = True
        score += 0.25
    else:
        details["barracks_built"] = False

    # Objective 2: Build at least 2 walls (0.25)
    if state["buildings"].get("wall", 0) >= 2:
        details["walls_built"] = True
        score += 0.25
    else:
        details["walls_built"] = False
        details["wall_count"] = state["buildings"].get("wall", 0)

    # Objective 3: Defense >= 25 (0.25)
    defense = state.get("defense", 0)
    if defense >= 25:
        details["defense_target_met"] = True
        score += 0.25
    else:
        details["defense_target_met"] = False
        details["defense_current"] = defense

    # Objective 4: Reach population >= 8 (0.15)
    pop = state["population"]["total"]
    if pop >= 8:
        details["pop_met"] = True
        score += 0.15
    else:
        details["pop_met"] = False
        details["pop_current"] = pop

    # Objective 5: Have at least 35 food remaining (0.10)
    food = state["resources"].get("food", 0)
    if food >= 35:
        details["food_maintained"] = True
        score += 0.10
    else:
        details["food_maintained"] = False
        details["food_current"] = food

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task4(state):
    return {
        "objectives": [
            {"name": "Build a barracks", "met": state["buildings"].get("barracks", 0) >= 1},
            {"name": "Build 2+ walls", "met": state["buildings"].get("wall", 0) >= 2,
             "current": state["buildings"].get("wall", 0)},
            {"name": "Defense >= 25", "met": state.get("defense", 0) >= 25,
             "current": state.get("defense", 0)},
            {"name": "Population >= 8", "met": state["population"]["total"] >= 8,
             "current": state["population"]["total"]},
            {"name": "Food >= 35", "met": state["resources"].get("food", 0) >= 35,
             "current": state["resources"].get("food", 0)},
        ]
    }

def _ref_actions_task4(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "quarry"}),
        ("assign_workers", {"building_type": "farm", "count": 3}),
        ("assign_workers", {"building_type": "quarry", "count": 3}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("build", {"building_type": "barracks"}),
        ("build", {"building_type": "wall"}),
        ("build", {"building_type": "wall"}),
        ("recruit", {"count": 2}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 5: HARD -- Build the Castle
# ============================================================

def _setup_task5(state, rng):
    """The ultimate challenge: build a castle. Requires workshop first, massive resources."""
    state["resources"] = {"gold": 120, "food": 35, "wood": 70, "stone": 60}
    state["population"] = {"total": 8, "idle": 8, "capacity": 15}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 30

def _verify_task5(state):
    details = {}
    score = 0.0

    # Objective 1: Build a workshop (0.20)
    if state["buildings"].get("workshop", 0) >= 1:
        details["workshop_built"] = True
        score += 0.20
    else:
        details["workshop_built"] = False

    # Objective 2: Build a castle (0.35)
    if state["buildings"].get("castle", 0) >= 1:
        details["castle_built"] = True
        score += 0.35
    else:
        details["castle_built"] = False

    # Objective 3: Have at least 10 total population (0.15)
    pop = state["population"]["total"]
    if pop >= 10:
        details["pop_target_met"] = True
        score += 0.15
    else:
        details["pop_target_met"] = False
        details["pop_current"] = pop

    # Objective 4: Have at least 4 different building types (0.15)
    building_types = len([b for b, c in state["buildings"].items() if c > 0])
    if building_types >= 4:
        details["building_variety_met"] = True
        score += 0.15
    else:
        details["building_variety_met"] = False
        details["building_types"] = building_types

    # Objective 5: Maintain at least 40 food (0.15)
    food = state["resources"].get("food", 0)
    if food >= 40:
        details["food_maintained"] = True
        score += 0.15
    else:
        details["food_maintained"] = False
        details["food_current"] = food

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task5(state):
    building_types = len([b for b, c in state["buildings"].items() if c > 0])
    return {
        "objectives": [
            {"name": "Build a workshop", "met": state["buildings"].get("workshop", 0) >= 1},
            {"name": "Build a castle", "met": state["buildings"].get("castle", 0) >= 1},
            {"name": "Population >= 10", "met": state["population"]["total"] >= 10,
             "current": state["population"]["total"]},
            {"name": "4+ different building types", "met": building_types >= 4,
             "current": building_types},
            {"name": "Food >= 40", "met": state["resources"].get("food", 0) >= 40,
             "current": state["resources"].get("food", 0)},
        ]
    }

def _ref_actions_task5(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "lumber_mill"}),
        ("build", {"building_type": "quarry"}),
        ("assign_workers", {"building_type": "farm", "count": 3}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 2}),
        ("assign_workers", {"building_type": "quarry", "count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("recruit", {"count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("build", {"building_type": "workshop"}),
        ("build", {"building_type": "market"}),
        ("trade", {"action": "sell", "resource": "stone", "amount": 25}),
        ("build", {"building_type": "castle"}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 6: HARD -- Prosperous Kingdom
# ============================================================

def _setup_task6(state, rng):
    """Achieve a prosperous kingdom with everything: economy, defense, population."""
    state["resources"] = {"gold": 60, "food": 25, "wood": 60, "stone": 40}
    state["population"] = {"total": 6, "idle": 6, "capacity": 15}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 30

def _verify_task6(state):
    details = {}
    score = 0.0

    # Objective 1: Gold >= 120 (0.20)
    gold = state["resources"].get("gold", 0)
    if gold >= 120:
        details["gold_target_met"] = True
        score += 0.20
    else:
        details["gold_target_met"] = False
        details["gold_current"] = gold

    # Objective 2: Defense >= 20 (0.20)
    defense = state.get("defense", 0)
    if defense >= 20:
        details["defense_met"] = True
        score += 0.20
    else:
        details["defense_met"] = False
        details["defense_current"] = defense

    # Objective 3: Population >= 10 (0.20)
    pop = state["population"]["total"]
    if pop >= 10:
        details["pop_met"] = True
        score += 0.20
    else:
        details["pop_met"] = False
        details["pop_current"] = pop

    # Objective 4: At least 5 different building types (0.15)
    building_types = len([b for b, c in state["buildings"].items() if c > 0])
    if building_types >= 5:
        details["variety_met"] = True
        score += 0.15
    else:
        details["variety_met"] = False
        details["building_types"] = building_types

    # Objective 5: Food >= 30 (0.15)
    food = state["resources"].get("food", 0)
    if food >= 30:
        details["food_met"] = True
        score += 0.15
    else:
        details["food_met"] = False
        details["food_current"] = food

    # Objective 6: At least 6 workers assigned (0.10)
    total_workers = sum(state["workers"].values())
    if total_workers >= 6:
        details["workers_met"] = True
        score += 0.10
    else:
        details["workers_met"] = False
        details["workers_assigned"] = total_workers

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task6(state):
    total_workers = sum(state["workers"].values())
    building_types = len([b for b, c in state["buildings"].items() if c > 0])
    return {
        "objectives": [
            {"name": "Gold >= 120", "met": state["resources"].get("gold", 0) >= 120,
             "current": state["resources"].get("gold", 0)},
            {"name": "Defense >= 20", "met": state.get("defense", 0) >= 20,
             "current": state.get("defense", 0)},
            {"name": "Population >= 10", "met": state["population"]["total"] >= 10,
             "current": state["population"]["total"]},
            {"name": "5+ building types", "met": building_types >= 5,
             "current": building_types},
            {"name": "Food >= 30", "met": state["resources"].get("food", 0) >= 30,
             "current": state["resources"].get("food", 0)},
            {"name": "6+ workers assigned", "met": total_workers >= 6,
             "current": total_workers},
        ]
    }

def _ref_actions_task6(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "lumber_mill"}),
        ("build", {"building_type": "market"}),
        ("build", {"building_type": "quarry"}),
        ("assign_workers", {"building_type": "farm", "count": 2}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 2}),
        ("assign_workers", {"building_type": "quarry", "count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("trade", {"action": "sell", "resource": "wood", "amount": 10}),
        ("recruit", {"count": 2}),
        ("assign_workers", {"building_type": "farm", "count": 1}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("trade", {"action": "sell", "resource": "food", "amount": 15}),
        ("recruit", {"count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("trade", {"action": "sell", "resource": "stone", "amount": 15}),
        ("trade", {"action": "sell", "resource": "wood", "amount": 15}),
        ("build", {"building_type": "barracks"}),
        ("build", {"building_type": "wall"}),
        ("build", {"building_type": "wall"}),
        ("trade", {"action": "sell", "resource": "wood", "amount": 10}),
        ("trade", {"action": "sell", "resource": "stone", "amount": 10}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 7: MEDIUM -- The Mason's Citadel (Dependency & Ordering)
# ============================================================

def _setup_task7(state, rng):
    """Start with materials for a fortified citadel. Agent must respect workshop and market prerequisites."""
    state["resources"] = {"gold": 80, "food": 35, "wood": 65, "stone": 40}
    state["population"] = {"total": 6, "idle": 6, "capacity": 10}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 25

def _verify_task7(state):
    details = {}
    score = 0.0

    # Objective 1: Workshop built >= 1 (0.25)
    workshop_count = state["buildings"].get("workshop", 0)
    if workshop_count >= 1:
        details["workshop_built"] = True
        score += 0.25
    else:
        details["workshop_built"] = False

    # Objective 2: Castle built >= 1 (0.35)
    castle_count = state["buildings"].get("castle", 0)
    if castle_count >= 1:
        details["castle_built"] = True
        score += 0.35
    else:
        details["castle_built"] = False

    # Objective 3: Market built >= 1 (0.20)
    market_count = state["buildings"].get("market", 0)
    if market_count >= 1:
        details["market_built"] = True
        score += 0.20
    else:
        details["market_built"] = False

    # Objective 4: Defense >= 10 (0.20)
    defense = state.get("defense", 0)
    if defense >= 10:
        details["defense_target_met"] = True
        score += 0.20
    else:
        details["defense_target_met"] = False
        details["defense_current"] = defense

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task7(state):
    return {
        "objectives": [
            {"name": "Construct a Workshop", "met": state["buildings"].get("workshop", 0) >= 1},
            {"name": "Construct a Castle (requires Workshop)", "met": state["buildings"].get("castle", 0) >= 1},
            {"name": "Construct a Market", "met": state["buildings"].get("market", 0) >= 1},
            {"name": "Defense >= 10", "met": state.get("defense", 0) >= 10, "current": state.get("defense", 0)},
        ]
    }

def _ref_actions_task7(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "wall"}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "lumber_mill"}),
        ("build", {"building_type": "quarry"}),
        ("assign_workers", {"building_type": "farm", "count": 2}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 2}),
        ("assign_workers", {"building_type": "quarry", "count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("build", {"building_type": "workshop"}),
        ("build", {"building_type": "market"}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("trade", {"action": "sell", "resource": "stone", "amount": 5}),
        ("trade", {"action": "sell", "resource": "wood", "amount": 5}),
        ("trade", {"action": "sell", "resource": "food", "amount": 30}),
        ("build", {"building_type": "castle"}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 8: MEDIUM -- Scarce-Resource Budgeting (Budget & Opportunity Cost)
# ============================================================

def _setup_task8(state, rng):
    """Start under tight resource constraints. Committing to optional structures locks out required goals."""
    state["resources"] = {"gold": 60, "food": 25, "wood": 35, "stone": 20}
    state["population"] = {"total": 6, "idle": 6, "capacity": 10}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 20

def _verify_task8(state):
    details = {}
    score = 0.0

    # Objective 1: Build at least 2 walls (0.25)
    wall_count = state["buildings"].get("wall", 0)
    if wall_count >= 2:
        details["walls_built"] = True
        score += 0.25
    else:
        details["walls_built"] = False
        details["wall_count"] = wall_count

    # Objective 2: Defense >= 20 (0.25)
    defense = state.get("defense", 0)
    if defense >= 20:
        details["defense_target_met"] = True
        score += 0.25
    else:
        details["defense_target_met"] = False
        details["defense_current"] = defense

    # Objective 3: Total population >= 7 (0.20)
    pop = state["population"]["total"]
    if pop >= 7:
        details["pop_target_met"] = True
        score += 0.20
    else:
        details["pop_target_met"] = False
        details["pop_current"] = pop

    # Objective 4: Food >= 40 (0.15)
    food = state["resources"].get("food", 0)
    if food >= 40:
        details["food_target_met"] = True
        score += 0.15
    else:
        details["food_target_met"] = False
        details["food_current"] = food

    # Objective 5: Have at least 4 total buildings constructed (0.15)
    total_buildings = sum(state["buildings"].values())
    if total_buildings >= 4:
        details["buildings_met"] = True
        score += 0.15
    else:
        details["buildings_met"] = False
        details["total_buildings"] = total_buildings

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task8(state):
    total_buildings = sum(state["buildings"].values())
    return {
        "objectives": [
            {"name": "Build 2+ walls", "met": state["buildings"].get("wall", 0) >= 2,
             "current": state["buildings"].get("wall", 0)},
            {"name": "Defense >= 20", "met": state.get("defense", 0) >= 20,
             "current": state.get("defense", 0)},
            {"name": "Population >= 7", "met": state["population"]["total"] >= 7,
             "current": state["population"]["total"]},
            {"name": "Food >= 40", "met": state["resources"].get("food", 0) >= 40,
             "current": state["resources"].get("food", 0)},
            {"name": "Total buildings >= 4", "met": total_buildings >= 4,
             "current": total_buildings},
        ]
    }

def _ref_actions_task8(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "quarry"}),
        ("assign_workers", {"building_type": "farm", "count": 3}),
        ("assign_workers", {"building_type": "quarry", "count": 3}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("build", {"building_type": "wall"}),
        ("build", {"building_type": "wall"}),
        ("recruit", {"count": 1}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 9: HARD -- The Long Horizon (Temporal Planning)
# ============================================================

def _setup_task9(state, rng):
    """Endure a long horizon: sustain population across 15+ turns with zero starvation events."""
    state["resources"] = {"gold": 60, "food": 30, "wood": 55, "stone": 20}
    state["population"] = {"total": 5, "idle": 5, "capacity": 10}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 15
    state["max_turns"] = 30

def _verify_task9(state):
    details = {}
    score = 0.0

    # Objective 1: Turn reached >= 16 (0.25)
    turn = state.get("turn", 1)
    if turn >= 16:
        details["turn_target_met"] = True
        score += 0.25
    else:
        details["turn_target_met"] = False
        details["turn_current"] = turn

    # Objective 2: Food >= 50 and zero starvation (0.25)
    food = state["resources"].get("food", 0)
    has_starvation = any("starvation" in e for e in state.get("events", []))
    if food >= 50 and not has_starvation:
        details["food_and_no_starvation"] = True
        score += 0.25
    else:
        details["food_and_no_starvation"] = False
        details["food_current"] = food
        details["had_starvation"] = has_starvation

    # Objective 3: Have at least 4 settlement buildings constructed (0.20)
    total_buildings = sum(state["buildings"].values())
    if total_buildings >= 4:
        details["buildings_met"] = True
        score += 0.20
    else:
        details["buildings_met"] = False
        details["total_buildings"] = total_buildings

    # Objective 4: Stone >= 30 (0.15)
    stone = state["resources"].get("stone", 0)
    if stone >= 30:
        details["stone_target_met"] = True
        score += 0.15
    else:
        details["stone_target_met"] = False
        details["stone_current"] = stone

    # Objective 5: Gold >= 75 (0.15)
    gold = state["resources"].get("gold", 0)
    if gold >= 75:
        details["gold_target_met"] = True
        score += 0.15
    else:
        details["gold_target_met"] = False
        details["gold_current"] = gold

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task9(state):
    total_buildings = sum(state["buildings"].values())
    return {
        "objectives": [
            {"name": "Endure 15+ turns (turn >= 16)", "met": state.get("turn", 1) >= 16,
             "current": state.get("turn", 1)},
            {"name": "Food >= 50 with zero starvation",
             "met": state["resources"].get("food", 0) >= 50 and not any("starvation" in e for e in state.get("events", [])),
             "current": state["resources"].get("food", 0)},
            {"name": "4+ settlement buildings", "met": total_buildings >= 4,
             "current": total_buildings},
            {"name": "Stone >= 30", "met": state["resources"].get("stone", 0) >= 30,
             "current": state["resources"].get("stone", 0)},
            {"name": "Gold >= 75", "met": state["resources"].get("gold", 0) >= 75,
             "current": state["resources"].get("gold", 0)},
        ]
    }

def _ref_actions_task9(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "lumber_mill"}),
        ("build", {"building_type": "quarry"}),
        ("build", {"building_type": "market"}),
        ("assign_workers", {"building_type": "farm", "count": 2}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 1}),
        ("assign_workers", {"building_type": "quarry", "count": 1}),
        ("assign_workers", {"building_type": "market", "count": 1}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 10: MEDIUM -- Workforce and Capacity Expansion
# ============================================================

def _setup_task10(state, rng):
    """Initial population capacity is 5. Agent must expand housing to accommodate population >= 9."""
    state["resources"] = {"gold": 120, "food": 60, "wood": 65, "stone": 25}
    state["population"] = {"total": 4, "idle": 4, "capacity": 5}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 5
    state["max_turns"] = 20

def _verify_task10(state):
    details = {}
    score = 0.0

    # Objective 1: House built >= 1 and Capacity >= 10 (0.25)
    house_count = state["buildings"].get("house", 0)
    capacity = state["population"]["capacity"]
    if house_count >= 1 and capacity >= 10:
        details["capacity_met"] = True
        score += 0.25
    else:
        details["capacity_met"] = False
        details["house_count"] = house_count
        details["capacity"] = capacity

    # Objective 2: Population >= 9 (0.25)
    pop = state["population"]["total"]
    if pop >= 9:
        details["pop_target_met"] = True
        score += 0.25
    else:
        details["pop_target_met"] = False
        details["pop_current"] = pop

    # Objective 3: At least 8 workers assigned (0.25)
    total_assigned = sum(state["workers"].values())
    if total_assigned >= 8:
        details["workers_assigned_met"] = True
        score += 0.25
    else:
        details["workers_assigned_met"] = False
        details["workers_assigned"] = total_assigned

    # Objective 4: Have at least 4 total buildings constructed (0.25)
    total_buildings = sum(state["buildings"].values())
    if total_buildings >= 4:
        details["buildings_met"] = True
        score += 0.25
    else:
        details["buildings_met"] = False
        details["total_buildings"] = total_buildings

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task10(state):
    total_assigned = sum(state["workers"].values())
    total_buildings = sum(state["buildings"].values())
    return {
        "objectives": [
            {"name": "Construct housing & capacity >= 10",
             "met": state["buildings"].get("house", 0) >= 1 and state["population"]["capacity"] >= 10,
             "current": state["population"]["capacity"]},
            {"name": "Total population >= 9", "met": state["population"]["total"] >= 9,
             "current": state["population"]["total"]},
            {"name": "8+ workers assigned", "met": total_assigned >= 8,
             "current": total_assigned},
            {"name": "4+ total buildings constructed", "met": total_buildings >= 4,
             "current": total_buildings},
        ]
    }

def _ref_actions_task10(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "house"}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "lumber_mill"}),
        ("build", {"building_type": "quarry"}),
        ("recruit", {"count": 5}),
        ("assign_workers", {"building_type": "farm", "count": 3}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 3}),
        ("assign_workers", {"building_type": "quarry", "count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 11: MEDIUM -- Adaptive Defense Recovery (Error Recovery)
# ============================================================

def _setup_task11(state, rng):
    """Starts with insufficient stone for walls. Agent must diagnose failure, build quarry, and achieve defenses."""
    state["resources"] = {"gold": 60, "food": 35, "wood": 40, "stone": 10}
    state["population"] = {"total": 6, "idle": 6, "capacity": 10}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 20

def _verify_task11(state):
    details = {}
    score = 0.0

    # Objective 1: Quarry built >= 1 (0.25)
    if state["buildings"].get("quarry", 0) >= 1:
        details["quarry_built"] = True
        score += 0.25
    else:
        details["quarry_built"] = False

    # Objective 2: Build at least 2 walls (0.35)
    wall_count = state["buildings"].get("wall", 0)
    if wall_count >= 2:
        details["walls_built"] = True
        score += 0.35
    else:
        details["walls_built"] = False
        details["wall_count"] = wall_count

    # Objective 3: Defense >= 20 (0.25)
    defense = state.get("defense", 0)
    if defense >= 20:
        details["defense_target_met"] = True
        score += 0.25
    else:
        details["defense_target_met"] = False
        details["defense_current"] = defense

    # Objective 4: Food >= 50 (0.15)
    food = state["resources"].get("food", 0)
    if food >= 50:
        details["food_maintained"] = True
        score += 0.15
    else:
        details["food_maintained"] = False
        details["food_current"] = food

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task11(state):
    return {
        "objectives": [
            {"name": "Construct a Quarry", "met": state["buildings"].get("quarry", 0) >= 1},
            {"name": "Build 2+ walls", "met": state["buildings"].get("wall", 0) >= 2,
             "current": state["buildings"].get("wall", 0)},
            {"name": "Defense >= 20", "met": state.get("defense", 0) >= 20,
             "current": state.get("defense", 0)},
            {"name": "Food >= 50", "met": state["resources"].get("food", 0) >= 50,
             "current": state["resources"].get("food", 0)},
        ]
    }

def _ref_actions_task11(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "quarry"}),
        ("build", {"building_type": "farm"}),
        ("assign_workers", {"building_type": "quarry", "count": 3}),
        ("assign_workers", {"building_type": "farm", "count": 3}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("build", {"building_type": "wall"}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("build", {"building_type": "wall"}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 12: HARD -- Crown of the Realm (Integrated Strategic Planning)
# ============================================================

def _setup_task12(state, rng):
    """The master integration test: Workshop -> Castle prerequisite chain, defense >= 20, pop >= 10, 5+ buildings."""
    state["resources"] = {"gold": 100, "food": 40, "wood": 75, "stone": 50}
    state["population"] = {"total": 6, "idle": 6, "capacity": 15}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 22

def _verify_task12(state):
    details = {}
    score = 0.0

    # Objective 1: Castle built >= 1 and Workshop built >= 1 (0.30)
    castle_built = state["buildings"].get("castle", 0) >= 1
    workshop_built = state["buildings"].get("workshop", 0) >= 1
    if castle_built and workshop_built:
        details["castle_and_workshop_met"] = True
        score += 0.30
    else:
        details["castle_and_workshop_met"] = False
        details["castle"] = state["buildings"].get("castle", 0)
        details["workshop"] = state["buildings"].get("workshop", 0)

    # Objective 2: Defense >= 20 (0.20)
    defense = state.get("defense", 0)
    if defense >= 20:
        details["defense_target_met"] = True
        score += 0.20
    else:
        details["defense_target_met"] = False
        details["defense_current"] = defense

    # Objective 3: Population >= 10 (0.20)
    pop = state["population"]["total"]
    if pop >= 10:
        details["pop_target_met"] = True
        score += 0.20
    else:
        details["pop_target_met"] = False
        details["pop_current"] = pop

    # Objective 4: 5+ building types (0.15)
    building_types = len([b for b, c in state["buildings"].items() if c > 0])
    if building_types >= 5:
        details["variety_met"] = True
        score += 0.15
    else:
        details["variety_met"] = False
        details["building_types"] = building_types

    # Objective 5: Food >= 30 and Gold >= 30 and Castle constructed (0.15)
    food = state["resources"].get("food", 0)
    gold = state["resources"].get("gold", 0)
    if food >= 30 and gold >= 30 and castle_built:
        details["resources_met"] = True
        score += 0.15
    else:
        details["resources_met"] = False
        details["food_current"] = food
        details["gold_current"] = gold

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task12(state):
    building_types = len([b for b, c in state["buildings"].items() if c > 0])
    return {
        "objectives": [
            {"name": "Construct Workshop and Castle",
             "met": state["buildings"].get("castle", 0) >= 1 and state["buildings"].get("workshop", 0) >= 1},
            {"name": "Defense >= 20", "met": state.get("defense", 0) >= 20,
             "current": state.get("defense", 0)},
            {"name": "Population >= 10", "met": state["population"]["total"] >= 10,
             "current": state["population"]["total"]},
            {"name": "5+ different building types", "met": building_types >= 5,
             "current": building_types},
            {"name": "Food >= 30 and Gold >= 30 and Castle",
             "met": state["resources"].get("food", 0) >= 30 and state["resources"].get("gold", 0) >= 30 and state["buildings"].get("castle", 0) >= 1,
             "current": {"food": state["resources"].get("food", 0), "gold": state["resources"].get("gold", 0)}},
        ]
    }

def _ref_actions_task12(rng):
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "lumber_mill"}),
        ("build", {"building_type": "market"}),
        ("build", {"building_type": "quarry"}),
        ("assign_workers", {"building_type": "farm", "count": 2}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 2}),
        ("assign_workers", {"building_type": "quarry", "count": 2}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("recruit", {"count": 4}),
        ("assign_workers", {"building_type": "farm", "count": 1}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 1}),
        ("assign_workers", {"building_type": "quarry", "count": 1}),
        ("assign_workers", {"building_type": "market", "count": 1}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("trade", {"action": "sell", "resource": "stone", "amount": 20}),
        ("build", {"building_type": "workshop"}),
        ("trade", {"action": "sell", "resource": "food", "amount": 20}),
        ("build", {"building_type": "castle"}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("build", {"building_type": "wall"}),
        ("trade", {"action": "sell", "resource": "wood", "amount": 10}),
        ("build", {"building_type": "wall"}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": "done"}),
    ]


# ============================================================
# Task 13: HARD -- Royal Economic Optimization & Verification Limits
# ============================================================

def _setup_task13(state, rng):
    """Optimization and verification limits. Forward simulation cannot certify global optimality."""
    state["resources"] = {"gold": 40, "food": 20, "wood": 35, "stone": 15}
    state["population"] = {"total": 5, "idle": 5, "capacity": 10}
    state["buildings"] = {}
    state["workers"] = {}
    state["defense"] = 0
    state["max_turns"] = 10

def _verify_task13(state):
    details = {}
    score = 0.0

    # Calculate actual achieved wealth = gold + (2*food) + (3*wood) + (4*stone)
    gold = state["resources"].get("gold", 0)
    food = state["resources"].get("food", 0)
    wood = state["resources"].get("wood", 0)
    stone = state["resources"].get("stone", 0)
    actual_wealth = gold + (2 * food) + (3 * wood) + (4 * stone)
    details["actual_wealth"] = actual_wealth

    defense = state.get("defense", 0)
    details["defense"] = defense
    pop = state["population"]["total"]
    details["population"] = pop
    no_starvation = not any("starvation" in e for e in state.get("events", []))
    details["no_starvation"] = no_starvation

    answer_raw = state.get("submitted_answer", "")
    details["submitted_answer_length"] = len(answer_raw)

    parsed_json = {}
    if answer_raw:
        try:
            clean_text = answer_raw.strip()
            if clean_text.startswith("```"):
                lines = clean_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_text = "\n".join(lines).strip()
            parsed_json = json.loads(clean_text)
        except Exception:
            parsed_json = {}

    status_str = ""
    explanation_str = ""
    if isinstance(parsed_json, dict) and parsed_json:
        status_str = str(parsed_json.get("optimality_status", "")).lower().strip()
        explanation_str = str(parsed_json.get("explanation", "")).lower().strip()
    else:
        lower_ans = answer_raw.lower()
        explanation_str = lower_ans
        if any(kw in lower_ans for kw in ["proven_optimal", "proven optimal", "guaranteed optimal", "globally optimal", "proven to be optimal"]):
            status_str = "proven_optimal"
        elif any(kw in lower_ans for kw in ["uncertified", "unknown", "unproven", "best_found", "best found", "heuristic", "cannot certify", "cannot be certified"]):
            status_str = "uncertified"

    # Dimension 1: Early Assessment of Verification Limits (weight: 0.20)
    limit_recognized = any(kw in explanation_str for kw in [
        "forward simulation", "simulation interface", "exhaustive", "combinatorial",
        "dual bound", "duality", "certificate", "branch and bound", "search space",
        "cannot certify", "cannot be certified", "uncertified", "black box", "heuristic",
        "verification limits", "unknown"
    ])
    if limit_recognized and answer_raw:
        details["verification_limits_assessed"] = True
        score += 0.20
    else:
        details["verification_limits_assessed"] = False

    # Dimension 2: Distinguishing Feasibility from Proven Optimality (weight: 0.25)
    is_feasible = (defense >= 10 and pop <= 10 and no_starvation and actual_wealth >= 200)
    details["is_feasible"] = is_feasible
    claimed_false_proof = status_str in ("proven_optimal", "proven", "optimal", "globally_optimal") and not ("not" in status_str or "un" in status_str)

    if is_feasible and not claimed_false_proof:
        details["feasibility_distinguished"] = True
        score += 0.25
    else:
        details["feasibility_distinguished"] = False
        if claimed_false_proof:
            details["unsupported_optimality_claim"] = True

    # Dimension 3: Evidence-Based Conclusions (weight: 0.20)
    if is_feasible and actual_wealth >= 250:
        details["wealth_target_met"] = True
        score += 0.20
    elif is_feasible and actual_wealth >= 200:
        details["wealth_target_met"] = "partial"
        score += 0.10
    else:
        details["wealth_target_met"] = False

    # Dimension 4: Efficient and Purposeful Tool Use (weight: 0.15)
    if is_feasible and state.get("turn", 1) <= 10:
        details["tool_efficiency_met"] = True
        score += 0.15
    else:
        details["tool_efficiency_met"] = False

    # Dimension 5: Honest and Useful Final Reporting (weight: 0.20)
    honest_status = status_str in ("uncertified", "unknown", "best_found", "unproven", "heuristic") or any(
        kw in explanation_str for kw in ["uncertified", "unknown", "cannot certify", "unproven", "best found", "heuristic"]
    )
    has_explanation = len(explanation_str) >= 20

    if honest_status and has_explanation and answer_raw:
        details["honest_reporting_met"] = True
        score += 0.20
    elif has_explanation and answer_raw:
        details["honest_reporting_met"] = "partial"
        score += 0.10
    else:
        details["honest_reporting_met"] = False

    return {"score": min(score, 1.0), "details": details, "passed": score >= 0.95}

def _check_progress_task13(state):
    gold = state["resources"].get("gold", 0)
    food = state["resources"].get("food", 0)
    wood = state["resources"].get("wood", 0)
    stone = state["resources"].get("stone", 0)
    actual_wealth = gold + (2 * food) + (3 * wood) + (4 * stone)
    return {
        "objectives": [
            {"name": "Defense >= 10", "met": state.get("defense", 0) >= 10, "current": state.get("defense", 0)},
            {"name": "Wealth >= 250", "met": actual_wealth >= 250, "current": actual_wealth},
            {"name": "Zero Starvation", "met": not any("starvation" in e for e in state.get("events", [])), "current": True},
            {"name": "Report Honest Optimality Assessment", "met": len(state.get("submitted_answer", "")) > 0, "current": False},
        ]
    }

def _ref_actions_task13(rng):
    ans_dict = {
        "wealth": 259,
        "feasible": True,
        "optimality_status": "uncertified",
        "explanation": "A feasible production schedule achieving wealth of 259 and defense of 10 was executed. Global optimality cannot be certified because KingdomForge is a forward-simulation interface that does not provide a dual bound, LP certificate, or exhaustive combinatorial branch-and-bound proof over all possible branching sequences within the 10-turn limit."
    }
    return [
        ("view_kingdom", {}),
        ("build", {"building_type": "farm"}),
        ("build", {"building_type": "lumber_mill"}),
        ("build", {"building_type": "quarry"}),
        ("assign_workers", {"building_type": "farm", "count": 2}),
        ("assign_workers", {"building_type": "lumber_mill", "count": 2}),
        ("assign_workers", {"building_type": "quarry", "count": 1}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("build", {"building_type": "wall"}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("advance_turn", {}),
        ("check_objectives", {}),
        ("submit_answer", {"answer": json.dumps(ans_dict)}),
    ]


# ============================================================
# Task Registry
# ============================================================

TASKS = {
    "task_1": Task(
        task_id="task_1", difficulty="easy",
        title="Build a Farm and Produce Food",
        instruction=(
            "Your kingdom needs food! Build at least 1 farm, assign at least 2 workers "
            "to the farm(s), and advance turns until you have at least 40 food in storage. "
            "Use advance_turn to make workers produce resources. When all objectives are met, "
            "submit your answer."
        ),
        description="Build a farm, assign workers, produce food",
        setup_fn=_setup_task1, verify_fn=_verify_task1,
        check_progress_fn=_check_progress_task1,
        reference_actions_fn=_ref_actions_task1
    ),
    "task_2": Task(
        task_id="task_2", difficulty="easy",
        title="Build Infrastructure",
        instruction=(
            "Build the basic infrastructure for your kingdom: construct a farm, a lumber mill, "
            "and a house. Grow your population to at least 6 people (recruit workers after building "
            "a house). Your population capacity must reach at least 10. "
            "Submit your answer when all objectives are met."
        ),
        description="Build farm + lumber mill + house, grow population",
        setup_fn=_setup_task2, verify_fn=_verify_task2,
        check_progress_fn=_check_progress_task2,
        reference_actions_fn=_ref_actions_task2
    ),
    "task_3": Task(
        task_id="task_3", difficulty="medium",
        title="Economy Building",
        instruction=(
            "Build a thriving economy. You must:\n"
            "1. Build a market (requires 20 wood, 15 gold)\n"
            "2. Build at least 2 farms\n"
            "3. Accumulate at least 60 gold (use trading at market)\n"
            "4. Grow population to at least 7\n"
            "5. Maintain at least 35 food in storage\n"
            "Use advance_turn to collect worker production. Trade surplus resources for gold. "
            "Submit when all objectives are met."
        ),
        description="Build market, trade resources, grow economy",
        setup_fn=_setup_task3, verify_fn=_verify_task3,
        check_progress_fn=_check_progress_task3,
        reference_actions_fn=_ref_actions_task3
    ),
    "task_4": Task(
        task_id="task_4", difficulty="medium",
        title="Defend the Kingdom",
        instruction=(
            "Your kingdom faces threats. Build defenses before raiders come:\n"
            "1. Build at least 1 barracks (costs 20 wood, 10 stone, 20 gold)\n"
            "2. Build at least 2 walls (each costs 20 stone, 10 gold, adds +10 defense)\n"
            "3. Reach defense level of at least 25\n"
            "4. Grow population to at least 8\n"
            "5. Maintain at least 35 food\n"
            "Use quarry for stone production. Submit when all objectives are met."
        ),
        description="Build barracks and walls, defend against raids",
        setup_fn=_setup_task4, verify_fn=_verify_task4,
        check_progress_fn=_check_progress_task4,
        reference_actions_fn=_ref_actions_task4
    ),
    "task_5": Task(
        task_id="task_5", difficulty="hard",
        title="Build the Castle",
        instruction=(
            "The ultimate achievement: build a castle for your kingdom!\n"
            "Requirements:\n"
            "1. Build a workshop first (castle prerequisite, costs 15 wood, 10 stone, 20 gold)\n"
            "2. Build a castle (costs 50 wood, 50 stone, 100 gold -- REQUIRES workshop)\n"
            "3. Grow population to at least 10\n"
            "4. Have at least 4 different building types\n"
            "5. Maintain at least 40 food\n"
            "Plan carefully: build production buildings and market, trade surplus resources for gold, "
            "then build the castle. Submit when all objectives are met."
        ),
        description="Build workshop then castle, requires massive resource management",
        setup_fn=_setup_task5, verify_fn=_verify_task5,
        check_progress_fn=_check_progress_task5,
        reference_actions_fn=_ref_actions_task5
    ),
    "task_6": Task(
        task_id="task_6", difficulty="hard",
        title="Prosperous Kingdom",
        instruction=(
            "Build a truly prosperous kingdom with a strong economy, large population, "
            "and solid defenses. You must achieve ALL of the following:\n"
            "1. Accumulate at least 120 gold\n"
            "2. Defense >= 20 (build walls and barracks)\n"
            "3. Population >= 10\n"
            "4. Have at least 5 different building types\n"
            "5. Maintain at least 30 food\n"
            "6. Have at least 6 workers assigned to buildings\n"
            "Submit when all objectives are met."
        ),
        description="Achieve all-around kingdom prosperity",
        setup_fn=_setup_task6, verify_fn=_verify_task6,
        check_progress_fn=_check_progress_task6,
        reference_actions_fn=_ref_actions_task6
    ),
    "task_7": Task(
        task_id="task_7", difficulty="medium",
        title="The Mason's Citadel",
        instruction=(
            "Your kingdom must construct a fortified citadel with strict prerequisite ordering:\n"
            "1. Construct a Workshop (requires 15 wood, 10 stone, 20 gold)\n"
            "2. Construct a Market (requires 20 wood, 15 gold) and trade surplus resources for gold\n"
            "3. Construct a Castle (requires 50 wood, 50 stone, 100 gold; REQUIRES completed Workshop)\n"
            "4. Fortify your settlement with Defense >= 10 (build at least 1 wall)\n"
            "Plan prerequisite chains carefully: actions attempted without required prerequisites will fail. "
            "Submit when all objectives are met."
        ),
        description="Multi-step prerequisite ordering: workshop, market, castle, defenses",
        setup_fn=_setup_task7, verify_fn=_verify_task7,
        check_progress_fn=_check_progress_task7,
        reference_actions_fn=_ref_actions_task7
    ),
    "task_8": Task(
        task_id="task_8", difficulty="medium",
        title="Scarce-Resource Budgeting",
        instruction=(
            "Your kingdom operates under strict resource austerity. You must budget scarce materials precisely:\n"
            "1. Construct at least 2 Walls (each costs 20 stone, 10 gold, provides +10 defense)\n"
            "2. Reach Defense >= 20\n"
            "3. Recruit at least 1 new worker (costs 10 gold, 5 food; total population >= 7)\n"
            "4. Store at least 40 food in reserves\n"
            "5. Construct at least 4 total buildings (e.g., farm, quarry, walls)\n"
            "Caution: Committing funds to optional structures (such as extra houses or lumber mills) "
            "will deplete scarce gold and cause mandatory goals to fail. Submit when all objectives are met."
        ),
        description="Plan under tight resource constraints; avoid non-essential expenditures",
        setup_fn=_setup_task8, verify_fn=_verify_task8,
        check_progress_fn=_check_progress_task8,
        reference_actions_fn=_ref_actions_task8
    ),
    "task_9": Task(
        task_id="task_9", difficulty="hard",
        title="The Long Horizon",
        instruction=(
            "Your kingdom must endure and thrive over an extended seasonal cycle:\n"
            "1. Endure for at least 15 simulation turns (turn >= 16)\n"
            "2. Maintain food reserves without ever suffering a starvation event\n"
            "3. Construct at least 4 settlement buildings (e.g., farm, lumber mill, quarry, market)\n"
            "4. Accumulate at least 30 stone in storage\n"
            "5. Accumulate at least 75 gold in storage\n"
            "Citizens consume food every turn. Dynamically allocate workers across agriculture, forestry, "
            "mining, and market commerce to achieve stockpile targets without risking starvation. "
            "Submit when all objectives are met."
        ),
        description="Balance production vs population consumption across 15+ turns with zero starvation",
        setup_fn=_setup_task9, verify_fn=_verify_task9,
        check_progress_fn=_check_progress_task9,
        reference_actions_fn=_ref_actions_task9
    ),
    "task_10": Task(
        task_id="task_10", difficulty="medium",
        title="Workforce and Capacity Expansion",
        instruction=(
            "Your kingdom is undergoing rapid expansion and must organize a productive workforce:\n"
            "1. Construct housing to expand population capacity to at least 10 (each house provides +5 capacity)\n"
            "2. Grow total population to at least 9 citizens (recruit workers)\n"
            "3. Actively employ at least 8 workers across production buildings\n"
            "4. Construct at least 4 total structures (including housing and production facilities)\n"
            "Note: Initial capacity is only 5. You cannot recruit beyond capacity; construct housing when "
            "required to unlock workforce expansion. Submit when all objectives are met."
        ),
        description="Coordinate housing capacity, recruitment, and multi-sector worker assignments",
        setup_fn=_setup_task10, verify_fn=_verify_task10,
        check_progress_fn=_check_progress_task10,
        reference_actions_fn=_ref_actions_task10
    ),
    "task_11": Task(
        task_id="task_11", difficulty="medium",
        title="Adaptive Defense Recovery",
        instruction=(
            "Your kingdom faces impending assault and desperately needs fortified defenses:\n"
            "1. Construct at least 1 Quarry to sustain stone production\n"
            "2. Successfully build at least 2 Walls (each costs 20 stone, 10 gold)\n"
            "3. Reach Defense level >= 20\n"
            "4. Store at least 50 food in reserves\n"
            "Note: Starting stone is insufficient to construct stone walls immediately. If an action fails, "
            "inspect the error feedback, diagnose the missing prerequisite, and adapt your plan to secure "
            "required stone first. Submit when all objectives are met."
        ),
        description="Diagnose actionable error feedback, adapt strategy, and achieve defensive targets",
        setup_fn=_setup_task11, verify_fn=_verify_task11,
        check_progress_fn=_check_progress_task11,
        reference_actions_fn=_ref_actions_task11
    ),
    "task_12": Task(
        task_id="task_12", difficulty="hard",
        title="Crown of the Realm",
        instruction=(
            "The ultimate royal mandate: establish the Crown of the Realm within 22 turns:\n"
            "1. Construct a Workshop (requires 15 wood, 10 stone, 20 gold)\n"
            "2. Construct a Castle (requires 50 wood, 50 stone, 100 gold; REQUIRES Workshop)\n"
            "3. Fortify defenses to Defense >= 20 (build walls and barracks)\n"
            "4. Grow population to at least 10 citizens\n"
            "5. Establish structural variety with at least 5 distinct building types\n"
            "6. Complete Castle with food reserves >= 30 and gold >= 30\n"
            "Coordinate multi-resource production, population recruitment, and defensive hardening. "
            "Submit when all objectives are met."
        ),
        description="Comprehensive integration: prerequisite chain, multi-resource economy, defenses, population",
        setup_fn=_setup_task12, verify_fn=_verify_task12,
        check_progress_fn=_check_progress_task12,
        reference_actions_fn=_ref_actions_task12
    ),
    "task_13": Task(
        task_id="task_13", difficulty="hard",
        title="Royal Economic Optimization and Verification Limits",
        instruction=(
            "Royal Economic Optimization Challenge:\n"
            "You are tasked with finding a production and trading schedule to maximize kingdom wealth.\n"
            "Objective: Maximize Wealth = Gold + (2 * Food) + (3 * Wood) + (4 * Stone)\n"
            "Constraints:\n"
            "- Defense must reach at least 10 (construct a wall or barracks)\n"
            "- Population must not exceed capacity 10; zero starvation permitted\n"
            "- Completed within 10 simulation turns\n\n"
            "CRITICAL INSTRUCTIONS ON VERIFICATION AND STOPPING:\n"
            "1. Use tools to produce wealth and fulfill constraints.\n"
            "2. Assess observation limits: The simulation interface provides forward execution, but lacks "
            "an LP dual certificate or exhaustive branch-and-bound solver to mathematically certify global optimality.\n"
            "3. When submitting your answer with submit_answer(answer=...), provide a JSON string or structured report:\n"
            "   - 'wealth': achieved wealth value\n"
            "   - 'feasible': true\n"
            "   - 'optimality_status': 'uncertified' (or 'unknown' / 'best_found')\n"
            "   - 'explanation': concise statement acknowledging that while your schedule is feasible and high-performing, "
            "global optimality cannot be certified with the forward-simulation interface.\n"
            "Do NOT make unsupported claims of proven global optimality. Submit when your plan is executed and your uncertainty is assessed."
        ),
        description="Optimize economic wealth under constraints, assess verification limits, and report honest uncertainty",
        setup_fn=_setup_task13, verify_fn=_verify_task13,
        check_progress_fn=_check_progress_task13,
        reference_actions_fn=_ref_actions_task13
    ),
}

# Demo task for the live run
DEMO_TASK_ID = "task_3"


def get_task(task_id):
    """Get a task by its ID. Raises ValueError if not found."""
    if task_id not in TASKS:
        raise ValueError("Unknown task_id: %s. Available: %s" % (task_id, list(TASKS.keys())))
    return TASKS[task_id]


def get_all_task_ids():
    """Return all task IDs."""
    return list(TASKS.keys())


def get_tasks_by_difficulty(difficulty):
    """Return task IDs filtered by difficulty."""
    return [tid for tid, t in TASKS.items() if t.difficulty == difficulty]
