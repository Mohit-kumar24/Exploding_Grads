# KingdomForge: Robust Reinforcement Learning Environment for World Simulation

KingdomForge is an autonomous agent reinforcement learning benchmark built for the **"Build an RL Environment That Can't Be Broken"** hackathon. It places autonomous LLM agents in control of a discrete-event medieval kingdom simulation where long-horizon resource management, multi-step construction planning, workforce allocation, and commercial trade are evaluated against strict programmatic verifiers.

---

## 1. Domain & Environment Design

KingdomForge models a discrete-event kingdom economy with persistent state and deterministic transitions:

- **Resources**: Gold, Food, Wood, and Stone.
- **Population**: Total citizens, idle workers, and housing capacity.
- **Infrastructure**: Farms, Lumber Mills, Quarries, Houses, Barracks, Markets, Walls, Workshops, and Castles.
- **Turn Dynamics**: Workers assigned to productive facilities generate yields upon turn advancement (`advance_turn`). Citizens consume 1 food unit per turn; running out of food triggers starvation mortality.
- **Commerce**: Markets unlock dynamic trading to sell surplus commodities or acquire deficient supplies.
- **Determinism**: Every state transition is strictly deterministic and parameterized by seed. Transitions generate a canonical 8-character state hash (`env.state_hash()`) to enable exact replay and auditing.

---

## 2. Benchmark Tasks & Baseline Gate Results

KingdomForge features **Benchmark v2**, consisting of 13 calibrated tasks across 3 difficulty tiers (Easy, Medium, Hard). Each task is empirically verified against the hackathon baseline gates:
- **Reference Agent Gate**: Must score $\ge 0.95$ (achieved **1.0000** on all 13 tasks).
- **No-Op Agent Gate**: Must score $\le 0.05$ (achieved **0.0000** on all 13 tasks).

Verified results recorded in [`baseline-results/gate.json`](baseline-results/gate.json):

| Task ID | Task Title | Difficulty | Max Turns | Ref Baseline | No-Op Baseline | Gate Check |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `task_1` | Resource Gathering | Easy | 15 turns | 1.0000 | 0.0000 | PASSED |
| `task_2` | Growing Hamlet | Easy | 20 turns | 1.0000 | 0.0000 | PASSED |
| `task_3` | Economic Engine (Demo) | Medium | 25 turns | 1.0000 | 0.0000 | PASSED |
| `task_4` | Defend the Kingdom | Medium | 25 turns | 1.0000 | 0.0000 | PASSED |
| `task_5` | Castle Construction | Hard | 30 turns | 1.0000 | 0.0000 | PASSED |
| `task_6` | Empire Builder | Hard | 30 turns | 1.0000 | 0.0000 | PASSED |
| `task_7` | The Mason's Citadel | Medium | 25 turns | 1.0000 | 0.0000 | PASSED |
| `task_8` | The Great Famine | Hard | 20 turns | 1.0000 | 0.0000 | PASSED |
| `task_9` | The Mercenary Ambush | Medium | 20 turns | 1.0000 | 0.0000 | PASSED |
| `task_10` | The Merchant Guild | Medium | 25 turns | 1.0000 | 0.0000 | PASSED |
| `task_11` | The Frozen Siege | Hard | 25 turns | 1.0000 | 0.0000 | PASSED |
| `task_12` | The Renaissance | Hard | 30 turns | 1.0000 | 0.0000 | PASSED |
| `task_13` | The Kingdom Audit | Hard | 25 turns | 1.0000 | 0.0000 | PASSED |

Total: **13 / 13 tasks passed** (100% gate compliance).

---

## 3. Architecture & Security Invariants

1. **Independent Programmatic Verification**: The agent cannot manipulate reward through linguistic persuasion or conversational hallucination. The [`Verifier`](src/verifier/verifier.py) queries internal simulation state snapshots directly.
2. **Strict Type & Argument Validation**: Boolean values passed to integer parameters (e.g. `count: true`) are explicitly rejected with parse errors. Negative quantities and unauthorized building types fail closed.
3. **Technological Prerequisites**: High-tier facilities require prerequisite infrastructure (e.g., Castles require Workshops; Markets require lumber mills and farms).
4. **Starvation Allocation Consistency**: Deaths deduct from idle workers first, then unassign workers from buildings, strictly preserving `idle + assigned == total`.
5. **Snapshot Isolation & Timeout Enforcement**: Tool calls execute in bounded threads (`TOOL_CALL_TIMEOUT = 60s`) against isolated state snapshots, preventing frozen calls from keeping the harness alive or corrupting live state.
6. **Credential Redaction**: Secrets (`API_KEY`, Bearer tokens, `sk-...`) are automatically sanitized from stdout, JSONL logs, and error records.

---

## 4. Configuration

The harness reads generic, model-agnostic configuration via `.env` or environment variables:

- `MODEL`: Model identifier (e.g., `gpt-4o-mini`, `luna-model`, or `mock`).
- `API_KEY`: API key for the chosen provider.
- `BASE_URL`: Base URL for the OpenAI-compatible endpoint.
- `TASK_ID`: Benchmark task to evaluate (default: `task_3`).
- `SEED`: Random seed for deterministic environment initialization (default: `42`).
- `MODE`: Execution mode — `full` (baselines + model), `baselines` (baselines only), or `all` (all tasks).
- `APPEND_FILE`: Instructions file appended to the system prompt (default: `inputs/append.txt`).

Copy the template:
```bash
cp .env.example .env
```

---

## 5. Execution Commands (Windows PowerShell)

Run all commands from the project root:
```powershell
# Navigate to the repository root
cd .
```

### A. Preflight Connectivity Check
Validates model connectivity, base URL, and credentials without spending episode tokens:
```powershell
python check_model.py
```

### B. Run Baseline Gates (All 13 Tasks)
Executes Reference and No-Op agents across all 13 benchmark tasks and updates `baseline-results/gate.json`:
```powershell
python run.py --mode baselines
```

### C. Run a Single Model Task (e.g. Task 3)
Runs baseline gates for the selected task, then invokes the model:
```powershell
python run.py --task task_3 --mode full
```

### D. Run Mock-Model Test Offline
Tests harness limits, conversation formatting, and JSONL logging without live API credentials:
```powershell
python run.py --task task_3 --model mock
```

### E. Run Full Test Suite (112 Tests)
Runs all 112 unit, security, invariant, limit, terminal formatting, and adversarial tests:
```powershell
python -m unittest discover tests
```

### F. Run Containerized Evaluation (Docker Compose)
Executes the live-run protocol inside Docker as specified by the hackathon evaluation rules:
```powershell
docker compose up --build --abort-on-container-exit --exit-code-from harness
```

---

## 6. Live Terminal Output & Telemetry

### Real-Time Terminal Visibility
The harness outputs live, flushed terminal updates:
- Selected model and endpoint (with API keys redacted).
- Step count (`[Step X/40]`) and model latency.
- Model response thought/text (cyan).
- Tool names and sanitized arguments (yellow).
- Tool execution outcomes and resulting state hashes.
- Clean ANSI colors on Windows PowerShell, Command Prompt, and Linux.
- Automatic fallback to plain unstyled text when `NO_COLOR=1` is set.

### Telemetry Logs
- Baseline gate summaries: [`baseline-results/gate.json`](baseline-results/gate.json)
- Curated genuine model runs: [`test-run-logs/`](test-run-logs/)
- Curated successful & failed trajectories: [`sample-trajectories/`](sample-trajectories/)
- Runtime episode logs: [`logs/`](logs/)

---

## 7. KingdomForge React Dashboard

A lightweight, local presentation dashboard is provided to inspect tasks, trajectories, test metrics, and baseline gate verifications.

To launch the dashboard:
```powershell
python scripts/serve_dashboard.py
```
Or open [`dashboard/index.html`](dashboard/index.html) directly in any modern browser.

Features:
- **Baseline Matrix**: Complete overview of all 13 tasks, difficulty ratings, turn budgets, and gate status.
- **Task Specification Explorer**: Detailed breakdown of objectives, setup conditions, and verifier criteria.
- **Interactive Trajectory Visualizer**: Side-by-side inspection of authentic successful and failed trajectories, showing step-by-step tool calls, arguments, and state hash progression.
- **Test Suite Metrics**: Live reporting of all 112 unit tests and pass rates.
- **PowerShell Guide**: Quick reference of all verified commands.

---

## 8. Hackathon Submission Artifacts

All required submission deliverables are generated, validated, and placed in their specified locations:

1. **`baseline-results/gate.json`**: Baseline gates for all 13 tasks (13 reference solutions @ 1.0000, 13 no-op baselines @ 0.0000).
2. **`test-run-logs/`**: Genuine model evaluation logs for Tasks 1 through 7 with step-by-step JSONL telemetry.
3. **`sample-trajectories/`**: Real sample trajectories including `success.jsonl` (Task 3 full success) and `failure.jsonl` (Task 7 genuine failure under strict prerequisites).
4. **`design-doc.pdf`**: 2-page system architecture, invariant specification, and benchmark design document.
5. **`deck.pdf`**: 10-slide comprehensive hackathon presentation deck.
6. **`kingdomforge_submission.zip`**: Complete submission archive containing all audited source code, tasks, tests, baselines, logs, trajectories, and documentation with zero secrets.

---

## 9. Repository Structure

```text
Hackathon/
|-- .env.example                     # Environment template (model, credentials, tasks)
|-- .gitignore                       # Git ignore protecting .env, logs, and artifacts
|-- AGENTS.md                        # Persistent agent workspace guidelines
|-- Dockerfile                       # Container definition for harness execution
|-- README.md                        # Primary documentation and benchmark specification
|-- check_model.py                   # Model connectivity and preflight verification script
|-- deck.pdf                         # 10-slide presentation deck (submission artifact)
|-- design-doc.pdf                   # 2-page system design document (submission artifact)
|-- docker-compose.yml               # Container orchestration matching hackathon spec
|-- kingdomforge_submission.zip      # Packaged submission archive (zero secrets, 79 files)
|-- requirements.txt                 # Python dependencies
|-- run.py                           # Top-level CLI entrypoint for evaluation harness
|-- test-results.json                # Cached test suite telemetry for dashboard API
|-- baseline-results/
|   `-- gate.json                    # Baseline gates (all 13 tasks: ref >= 0.95, noop <= 0.05)
|-- dashboard/                       # React presentation dashboard
|   |-- index.html                   # HTML entrypoint
|   |-- dashboard_app.js             # Compiled vanilla React application
|   `-- vendor/                      # Offline React, ReactDOM, Babel, and Tailwind CDN
|-- docs/                            # In-depth architectural documentation
|   |-- ARCHITECTURE.md              # System design and simulation invariants
|   |-- IMPLEMENTATION_PLAN.md       # Implementation milestones and phases
|   |-- REQUIREMENTS_CHECKLIST.md    # Requirement traceability matrix
|   |-- SPECIFICATION_AUDIT.md       # Specification compliance verification
|   |-- TASKS_BENCHMARK_V2.md        # Extended 13-task benchmark specifications
|   |-- project/                     # Project tracking and handoff records
|   |   |-- HANDOFF.md               # Operational handoff and status
|   |   |-- PROJECT_STATE.md         # Detailed milestone and system state
|   |   `-- TASKS.md                 # Task implementation tracking checklist
|   `-- spec/                        # Problem statement and reference extracts
|       |-- pdf_extract_primary.txt
|       |-- pdf_extract_starter.txt
|       `-- rl_hackathon_problem_statement.md
|-- inputs/
|   `-- append.txt                   # Prompt instruction file appended to system prompt
|-- logs/                            # Local ephemeral evaluation logs (gitignored)
|-- sample-trajectories/             # Real success and failure trajectories
|   |-- failure.jsonl
|   |-- failure_trajectory_task_3.json
|   |-- failure_trajectory_task_7.json
|   |-- success.jsonl
|   |-- success_trajectory_task3.json
|   `-- success_trajectory_task_3.json
|-- scripts/                         # Build, presentation, and packaging utilities
|   |-- generate_deck_pdf.py         # 10-slide PDF deck generator
|   |-- generate_design_doc_pdf.py   # 2-page PDF design doc generator
|   |-- package_submission.py        # Submission validation and zip packager
|   `-- serve_dashboard.py           # Local server for the React presentation dashboard
|-- src/                             # Core environment and harness source code
|   |-- agents/                      # Baseline, reference, and model agents
|   |   |-- baseline_agents.py       # No-op baseline agent
|   |   |-- reference_agent.py       # Deterministic reference solution agent (Tasks 1-13)
|   |   `-- tool_call_agent.py       # Model agent communicating with OpenAI-compatible API
|   |-- env/                         # Simulation environment and mechanics
|   |   |-- database_env.py          # SQLite-backed state machine and invariants
|   |   |-- tasks.py                 # Task definitions for Tasks 1-13
|   |   `-- tools.py                 # Tool schemas and definitions
|   |-- harness/                     # Evaluation orchestration and telemetry
|   |   |-- dashboard_data.py        # Data provider for presentation dashboard
|   |   |-- logger.py                # Telemetry logger with required step schema
|   |   |-- model_client.py          # Resilient OpenAI-compatible model client
|   |   |-- runner.py                # Orchestrator enforcing 40-step/10-min/10-tool limits
|   |   `-- terminal.py              # Windows cp1252-safe ANSI terminal formatter
|   `-- verifier/                    # Independent ground-truth verifiers
|       `-- verifier.py              # Programmatic state verifier and failure taxonomy
|-- test-run-logs/                   # Genuine model evaluation logs for Tasks 1-7
`-- tests/                           # 15 test suites (112 tests, 100% passing)


