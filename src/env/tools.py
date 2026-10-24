"""
KingdomForge Tool Definitions

Each tool has a name, description, and JSON-schema argument definition.
Tools are the agent's interface to the kingdom simulation.
"""

# All available tools as OpenAI-compatible function schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "view_kingdom",
            "description": "View the current state of your kingdom, including resources, population, buildings, and active workers.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "build",
            "description": "Construct a building in the kingdom. Buildings cost resources and take effect immediately once built. Building costs and effects: farm (10 wood, 5 gold; produces food, cap 3 workers); lumber_mill (5 wood, 10 gold; produces wood, cap 3 workers); quarry (15 wood, 15 gold; produces stone, cap 3 workers); house (15 wood, 10 gold; +5 population capacity); barracks (20 wood, 10 stone, 20 gold; +5 defense); market (20 wood, 15 gold; enables trade and produces gold, cap 2 workers); granary (15 wood, 5 stone; +50 food storage cap); wall (20 stone, 10 gold; +10 defense); workshop (15 wood, 10 stone, 20 gold; prerequisite for castle); castle (50 wood, 50 stone, 100 gold; requires workshop).",
            "parameters": {
                "type": "object",
                "properties": {
                    "building_type": {
                        "type": "string",
                        "enum": ["farm", "lumber_mill", "quarry", "house", "barracks",
                                 "market", "granary", "wall", "workshop", "castle"],
                        "description": "Type of building to construct"
                    }
                },
                "required": ["building_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assign_workers",
            "description": "Assign idle workers to a production building to generate resources each turn. Worker capacity per building: farm=3, lumber_mill=3, quarry=3, market=2 (multiple buildings multiply capacity). Valid assignments: farm (produces food), lumber_mill (produces wood), quarry (produces stone), market (produces gold).",
            "parameters": {
                "type": "object",
                "properties": {
                    "building_type": {
                        "type": "string",
                        "enum": ["farm", "lumber_mill", "quarry", "market"],
                        "description": "Building to assign workers to"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of workers to assign (must be positive)",
                        "minimum": 1
                    }
                },
                "required": ["building_type", "count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unassign_workers",
            "description": "Remove workers from a building and return them to the idle pool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "building_type": {
                        "type": "string",
                        "enum": ["farm", "lumber_mill", "quarry", "market"],
                        "description": "Building to remove workers from"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of workers to unassign",
                        "minimum": 1
                    }
                },
                "required": ["building_type", "count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trade",
            "description": "Trade resources at the market. You must have a market building. Exchange rates: sell_resource -> gold at varying rates, buy_resource with gold at varying rates. sell: food=2g, wood=3g, stone=4g per unit. buy: food=3g, wood=5g, stone=7g per unit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Whether to buy or sell resources"
                    },
                    "resource": {
                        "type": "string",
                        "enum": ["food", "wood", "stone"],
                        "description": "Resource to trade"
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Amount to trade (must be positive)",
                        "minimum": 1
                    }
                },
                "required": ["action", "resource", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recruit",
            "description": "Recruit new workers from the population into the idle pool. Requires available population capacity (check existing capacity before building houses; houses add +5 capacity). Each recruitment costs 10 gold and 5 food.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of workers to recruit",
                        "minimum": 1
                    }
                },
                "required": ["count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "advance_turn",
            "description": "End the current turn and advance to the next one. Assigned workers produce resources and population consumes food (1 food per person). Note: advancing turns does NOT produce gold unless a market is built and functioning.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_objectives",
            "description": "View the task objectives and your current progress toward each one. Call this to verify completion before submit_answer.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit your answer to finish the task. When check_objectives shows all objectives are met, call this immediately to conclude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Completion message or status"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]

# Tool name set for validation
VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


def get_tool_schemas():
    """Return a copy of tool schemas for the agent."""
    import copy
    return copy.deepcopy(TOOL_SCHEMAS)
