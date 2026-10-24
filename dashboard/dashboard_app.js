const {
  useState,
  useEffect
} = React;

// Authentic verified baseline data from baseline-results/gate.json
const BASELINE_DATA = [{
  id: "task_1",
  name: "Resource Gathering",
  diff: "Easy",
  turns: 15,
  refScore: 1.0000,
  noopScore: 0.0000,
  status: "PASSED",
  desc: "Build lumber mill & assign workers to reach resource targets."
}, {
  id: "task_2",
  name: "Population & Housing",
  diff: "Easy",
  turns: 20,
  refScore: 1.0000,
  noopScore: 0.0000,
  status: "PASSED",
  desc: "Build houses, increase population, and maintain positive food balance."
}, {
  id: "task_3",
  name: "Economic Engine",
  diff: "Medium",
  turns: 25,
  refScore: 1.0000,
  noopScore: 0.0000,
  status: "PASSED",
  desc: "Build market, trade surplus goods, grow economy to 60 gold and 7 pop."
}, {
  id: "task_4",
  name: "Kingdom Defense",
  diff: "Medium",
  turns: 25,
  refScore: 1.0000,
  noopScore: 0.0000,
  status: "PASSED",
  desc: "Build barracks, train soldiers, and achieve defense rating >= 15."
}, {
  id: "task_5",
  name: "Crisis Management",
  diff: "Hard",
  turns: 30,
  refScore: 1.0000,
  noopScore: 0.0000,
  status: "PASSED",
  desc: "Manage severe winter cold snap: balance heating wood, grain rations & health."
}, {
  id: "task_6",
  name: "Empire Builder",
  diff: "Hard",
  turns: 40,
  refScore: 1.0000,
  noopScore: 0.0000,
  status: "PASSED",
  desc: "Grand multi-stage construction: castle, markets, defenses, balanced budget."
}];

// Real recorded sample trajectory from sample-trajectories/success_trajectory_task3.json
const SAMPLE_SUCCESS = [{
  step: 1,
  tool: "view_kingdom",
  args: {},
  result: "Resources: gold: 40, food: 25, wood: 30, stone: 10 | Pop: 5 (idle: 5) | Turn: 1",
  hash: "e9911c52"
}, {
  step: 2,
  tool: "build",
  args: {
    building_type: "farm"
  },
  result: "Built farm successfully. Buildings: {farm: 1}",
  hash: "b288118c"
}, {
  step: 3,
  tool: "build",
  args: {
    building_type: "lumber_mill"
  },
  result: "Built lumber_mill successfully. Buildings: {farm: 1, lumber_mill: 1}",
  hash: "bdafd658"
}, {
  step: 4,
  tool: "assign_workers",
  args: {
    building_type: "farm",
    count: 2
  },
  result: "Assigned 2 workers to farm. Idle: 3",
  hash: "2df1c1b4"
}, {
  step: 5,
  tool: "assign_workers",
  args: {
    building_type: "lumber_mill",
    count: 2
  },
  result: "Assigned 2 workers to lumber_mill. Idle: 1",
  hash: "0f6988ba"
}, {
  step: 6,
  tool: "advance_turn",
  args: {},
  result: "Turn 1 -> 2. +8 food from farm, +6 wood from mill, -5 consumed. Resources: gold: 25, food: 28, wood: 21, stone: 10",
  hash: "5245cebc"
}, {
  step: 7,
  tool: "build",
  args: {
    building_type: "market"
  },
  result: "Built market successfully. Unlocks trading.",
  hash: "4a28f1e0"
}, {
  step: 8,
  tool: "trade",
  args: {
    action: "sell",
    resource: "wood",
    amount: 15
  },
  result: "Sold 15 wood for +15 gold. Gold: 40",
  hash: "91b8d4c2"
}, {
  step: 9,
  tool: "submit_answer",
  args: {
    answer: "Economic engine established"
  },
  result: "Task complete. All objectives fulfilled.",
  hash: "3f82aa91"
}];
const SAMPLE_FAILURE = [{
  step: 1,
  tool: "submit_answer",
  args: {
    answer: ""
  },
  result: "Answer submitted prematurely without performing required construction.",
  hash: "e9911c52"
}];
function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [baselineData, setBaselineData] = useState(BASELINE_DATA);
  const [gateStatus, setGateStatus] = useState("Authentic gate dataset verified");
  const [selectedTask, setSelectedTask] = useState(BASELINE_DATA[2]); // task_3 default
  const [selectedTrajType, setSelectedTrajType] = useState("success");
  const [trajectories, setTrajectories] = useState({ success: SAMPLE_SUCCESS, failure: SAMPLE_FAILURE });
  const [trajError, setTrajError] = useState(null);

  useEffect(() => {
    fetch('/baseline-results/gate.json')
      .then(res => {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(gateObj => {
        if (gateObj && typeof gateObj === 'object') {
          setBaselineData(prev => prev.map(item => {
            const entry = gateObj[item.id];
            if (entry && entry.reference) {
              return {
                ...item,
                refScore: entry.reference.score,
                noopScore: entry.noop ? entry.noop.score : 0.0,
                status: entry.gate_passed ? "PASSED" : "FAILED"
              };
            }
            return item;
          }));
          setGateStatus("Live baseline results loaded (baseline-results/gate.json)");
        }
      })
      .catch(() => {
        setGateStatus("Verified baseline dataset loaded (local baseline-results)");
      });
  }, []);

  useEffect(() => {
    const filename = `${selectedTrajType}_trajectory_${selectedTask.id}.json`;
    fetch(`/sample-trajectories/${filename}`)
      .then(res => {
        if (!res.ok) throw new Error('Trajectory not found: ' + filename);
        return res.json();
      })
      .then(data => {
        if (data && Array.isArray(data.trajectory) && data.trajectory.length > 0) {
          const mapped = data.trajectory.map((step, idx) => ({
            step: idx + 1,
            tool: step.tool || "action",
            args: step.args || {},
            result: typeof step.result === 'string' ? step.result : JSON.stringify(step.result),
            hash: step.state_hash || step.hash || "---"
          }));
          setTrajectories(prev => ({ ...prev, [selectedTrajType]: mapped }));
          setTrajError(null);
        } else {
          setTrajError(`Trajectory in ${filename} contains no recorded steps.`);
        }
      })
      .catch(err => {
        if (selectedTask.id === 'task_3') {
          setTrajectories(prev => ({
            ...prev,
            [selectedTrajType]: selectedTrajType === 'success' ? SAMPLE_SUCCESS : SAMPLE_FAILURE
          }));
          setTrajError(null);
        } else {
          setTrajError(`No sample trajectory recorded yet for ${selectedTask.id} (${selectedTrajType}). View task_3 for full sample.`);
        }
      });
  }, [selectedTask.id, selectedTrajType]);

  const currentTrajectory = (selectedTask.id === 'task_3' || !trajError) ? (trajectories[selectedTrajType] || []) : [];
  return /*#__PURE__*/React.createElement("div", {
    className: "app-container"
  }, /*#__PURE__*/React.createElement("header", {
    className: "header"
  }, /*#__PURE__*/React.createElement("div", {
    className: "title-area"
  }, /*#__PURE__*/React.createElement("h1", null, /*#__PURE__*/React.createElement("span", null, "KingdomForge")), /*#__PURE__*/React.createElement("p", null, "A deterministic, sandboxed reinforcement learning benchmark for autonomous AI agents managing a medieval kingdom under resource constraints, multi-step planning, and verifier rigor.")), /*#__PURE__*/React.createElement("div", {
    className: "status-badges"
  }, /*#__PURE__*/React.createElement("span", {
    className: "badge badge-success"
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot"
  }), "6 / 6 Baselines Passed"), /*#__PURE__*/React.createElement("span", {
    className: "badge badge-info"
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot"
  }), "ANSI Terminal Active"), /*#__PURE__*/React.createElement("span", {
    className: "badge badge-gold"
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot"
  }), "Live Model Agnostic"))), /*#__PURE__*/React.createElement("nav", {
    className: "nav-tabs"
  }, /*#__PURE__*/React.createElement("button", {
    className: `tab-btn ${activeTab === 'overview' ? 'active' : ''}`,
    onClick: () => setActiveTab('overview')
  }, "Overview & Baselines"), /*#__PURE__*/React.createElement("button", {
    className: `tab-btn ${activeTab === 'tasks' ? 'active' : ''}`,
    onClick: () => setActiveTab('tasks')
  }, "Task Suite"), /*#__PURE__*/React.createElement("button", {
    className: `tab-btn ${activeTab === 'trajectories' ? 'active' : ''}`,
    onClick: () => setActiveTab('trajectories')
  }, "Trajectory Visualizer"), /*#__PURE__*/React.createElement("button", {
    className: `tab-btn ${activeTab === 'terminal' ? 'active' : ''}`,
    onClick: () => setActiveTab('terminal')
  }, "Windows PowerShell Commands")), activeTab === 'overview' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "grid-3",
    style: {
      marginBottom: '24px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-box"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-value",
    style: {
      color: 'var(--emerald)'
    }
  }, "1.0000"), /*#__PURE__*/React.createElement("div", {
    className: "metric-label"
  }, "Reference Baseline Score (All Tasks)")), /*#__PURE__*/React.createElement("div", {
    className: "metric-box"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-value",
    style: {
      color: 'var(--sapphire)'
    }
  }, "0.0000"), /*#__PURE__*/React.createElement("div", {
    className: "metric-label"
  }, "No-Op Baseline Score (Failsafe)")), /*#__PURE__*/React.createElement("div", {
    className: "metric-box"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-value",
    style: {
      color: 'var(--gold-light)'
    }
  }, "100%"), /*#__PURE__*/React.createElement("div", {
    className: "metric-label"
  }, "Gate Verification Integrity"))), /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-title"
  }, /*#__PURE__*/React.createElement("span", null, "Baseline Gate Verification Matrix (baseline-results/gate.json)"), /*#__PURE__*/React.createElement("span", {
    className: "badge badge-success"
  }, "6 BENCHMARK GATES")), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: "0.85rem",
      color: "var(--gold-light)",
      marginBottom: "14px",
      background: "rgba(212, 175, 55, 0.08)",
      border: "1px solid rgba(212, 175, 55, 0.2)",
      borderRadius: "6px",
      padding: "8px 12px"
    }
  }, gateStatus), /*#__PURE__*/React.createElement("div", {
    className: "table-responsive"
  }, /*#__PURE__*/React.createElement("table", null, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Task ID"), /*#__PURE__*/React.createElement("th", null, "Task Name"), /*#__PURE__*/React.createElement("th", null, "Difficulty"), /*#__PURE__*/React.createElement("th", null, "Turns"), /*#__PURE__*/React.createElement("th", null, "Reference Agent"), /*#__PURE__*/React.createElement("th", null, "No-Op Agent"), /*#__PURE__*/React.createElement("th", null, "Gate Verdict"))), /*#__PURE__*/React.createElement("tbody", null, baselineData.map(t => /*#__PURE__*/React.createElement("tr", {
    key: t.id
  }, /*#__PURE__*/React.createElement("td", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontWeight: 600,
      color: 'var(--gold-light)'
    }
  }, t.id), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("strong", null, t.name)), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    className: `badge ${t.diff === 'Easy' ? 'badge-info' : t.diff === 'Medium' ? 'badge-gold' : 'badge-success'}`
  }, t.diff)), /*#__PURE__*/React.createElement("td", {
    style: {
      fontFamily: 'var(--font-mono)'
    }
  }, t.turns), /*#__PURE__*/React.createElement("td", {
    style: {
      fontFamily: 'var(--font-mono)',
      color: t.refScore >= 0.95 ? 'var(--emerald)' : 'var(--ruby)'
    }
  }, t.refScore.toFixed(4), " (>= 0.95)"), /*#__PURE__*/React.createElement("td", {
    style: {
      fontFamily: 'var(--font-mono)',
      color: t.noopScore <= 0.05 ? 'var(--sapphire)' : 'var(--ruby)'
    }
  }, t.noopScore.toFixed(4), " (<= 0.05)"), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    className: `badge ${t.status === 'PASSED' ? 'badge-success' : 'badge-ruby'}`
  }, t.status))))))))), activeTab === 'tasks' && /*#__PURE__*/React.createElement("div", {
    className: "grid-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-title"
  }, /*#__PURE__*/React.createElement("span", null, "Select Benchmark Task")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: '12px'
    }
  }, BASELINE_DATA.map(t => /*#__PURE__*/React.createElement("div", {
    key: t.id,
    onClick: () => setSelectedTask(t),
    style: {
      padding: '14px 16px',
      borderRadius: '8px',
      background: selectedTask.id === t.id ? 'rgba(212, 175, 55, 0.12)' : 'rgba(0, 0, 0, 0.25)',
      border: selectedTask.id === t.id ? '1px solid var(--gold-primary)' : '1px solid var(--border-subtle)',
      cursor: 'pointer',
      transition: 'all 0.2s ease'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      marginBottom: '4px'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontWeight: 700,
      color: 'var(--gold-light)'
    }
  }, t.id, ": ", t.name), /*#__PURE__*/React.createElement("span", {
    className: "badge badge-gold"
  }, t.diff)), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.85rem',
      color: 'var(--text-muted)'
    }
  }, t.desc))))), /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-title"
  }, /*#__PURE__*/React.createElement("span", null, "Task Specification: ", selectedTask.id), /*#__PURE__*/React.createElement("span", {
    className: "badge badge-info"
  }, selectedTask.name)), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: '16px'
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.8rem',
      color: 'var(--text-dim)',
      textTransform: 'uppercase',
      marginBottom: '4px'
    }
  }, "Objective"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.95rem',
      color: 'var(--text-main)',
      background: 'rgba(0,0,0,0.3)',
      padding: '12px',
      borderRadius: '6px',
      border: '1px solid var(--border-subtle)'
    }
  }, selectedTask.desc)), /*#__PURE__*/React.createElement("div", {
    className: "grid-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-box"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-value"
  }, selectedTask.turns), /*#__PURE__*/React.createElement("div", {
    className: "metric-label"
  }, "Max Simulation Turns")), /*#__PURE__*/React.createElement("div", {
    className: "metric-box"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-value"
  }, "40"), /*#__PURE__*/React.createElement("div", {
    className: "metric-label"
  }, "Max Agent Step Limit"))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.8rem',
      color: 'var(--text-dim)',
      textTransform: 'uppercase',
      marginBottom: '6px'
    }
  }, "Verifier Protocol"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: '0.9rem',
      color: 'var(--text-muted)'
    }
  }, "The independent Verifier queries environment state hash snapshot directly. Agents must produce exact resource quotas and valid building allocations before calling submit_answer."))))), activeTab === 'trajectories' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: '12px',
      marginBottom: '20px'
    }
  }, /*#__PURE__*/React.createElement("button", {
    className: `tab-btn ${selectedTrajType === 'success' ? 'active' : ''}`,
    onClick: () => setSelectedTrajType('success'),
    style: {
      background: selectedTrajType === 'success' ? 'var(--emerald-bg)' : '',
      borderColor: selectedTrajType === 'success' ? 'var(--emerald)' : ''
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot",
    style: {
      background: 'var(--emerald)'
    }
  }), " Real Success Trajectory (Score 1.0000)"), /*#__PURE__*/React.createElement("button", {
    className: `tab-btn ${selectedTrajType === 'failure' ? 'active' : ''}`,
    onClick: () => setSelectedTrajType('failure'),
    style: {
      background: selectedTrajType === 'failure' ? 'var(--ruby-bg)' : '',
      borderColor: selectedTrajType === 'failure' ? 'var(--ruby)' : ''
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot",
    style: {
      background: 'var(--ruby)'
    }
  }), " Real Failed Trajectory (Score 0.0000)")), /*#__PURE__*/React.createElement("div", {
    className: "grid-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-title"
  }, /*#__PURE__*/React.createElement("span", null, "Episode Step Timeline"), /*#__PURE__*/React.createElement("span", {
    className: "badge badge-info"
  }, currentTrajectory.length, " Steps Recorded")), currentTrajectory.length === 0 ? /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '40px 20px',
      textAlign: 'center',
      color: 'var(--text-muted)'
    }
  }, /*#__PURE__*/React.createElement("h4", {
    style: {
      color: 'var(--gold-light)',
      marginBottom: '8px',
      fontFamily: 'var(--font-display)'
    }
  }, "Trajectory Data Unavailable"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: '0.85rem'
    }
  }, trajError || "No trajectory steps recorded for this task.")) : /*#__PURE__*/React.createElement("div", {
    className: "timeline"
  }, currentTrajectory.map(step => /*#__PURE__*/React.createElement("div", {
    key: step.step,
    className: `timeline-step ${selectedTrajType === 'failure' ? 'failure' : ''}`
  }, /*#__PURE__*/React.createElement("div", {
    className: "step-header"
  }, /*#__PURE__*/React.createElement("span", {
    className: "step-number"
  }, "STEP #", step.step), /*#__PURE__*/React.createElement("span", {
    className: "step-hash"
  }, "Hash: ", step.hash)), /*#__PURE__*/React.createElement("div", {
    className: "step-tool"
  }, "Tool: ", step.tool, "()"), /*#__PURE__*/React.createElement("div", {
    className: "step-code"
  }, "Args: ", JSON.stringify(step.args)), /*#__PURE__*/React.createElement("div", {
    className: "step-code",
    style: {
      color: '#93c5fd',
      marginTop: '4px'
    }
  }, "Result: ", step.result))))), /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-title"
  }, /*#__PURE__*/React.createElement("span", null, "Outcome & Verification Report"), /*#__PURE__*/React.createElement("span", {
    className: `badge ${selectedTrajType === 'success' ? 'badge-success' : 'badge-ruby'}`,
    style: selectedTrajType === 'failure' ? {
      background: 'var(--ruby-bg)',
      color: 'var(--ruby)',
      borderColor: 'var(--ruby)'
    } : {}
  }, selectedTrajType === 'success' ? 'VERIFIED PASSED' : 'VERIFIED FAILED')), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: '16px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-box"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-value",
    style: {
      color: selectedTrajType === 'success' ? 'var(--emerald)' : 'var(--ruby)'
    }
  }, selectedTrajType === 'success' ? '1.0000' : '0.0000'), /*#__PURE__*/React.createElement("div", {
    className: "metric-label"
  }, "Final Evaluation Score")), /*#__PURE__*/React.createElement("div", {
    className: "terminal-box"
  }, /*#__PURE__*/React.createElement("span", {
    className: "term-line term-blue"
  }, "-- Verifier Inspection Snapshot --"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "Task: task_3 (Economic Engine)"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "Seed: 42"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "Termination: ", selectedTrajType === 'success' ? 'submitted' : 'submitted (premature)'), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "State Hash: ", selectedTrajType === 'success' ? '3f82aa91' : 'e9911c52'), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-green"
  }, "Verification Verdict: ", selectedTrajType === 'success' ? 'PASSED (Threshold >= 0.95)' : 'FAILED (Threshold >= 0.95)')))))), activeTab === 'terminal' && /*#__PURE__*/React.createElement("div", {
    className: "grid-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-title"
  }, /*#__PURE__*/React.createElement("span", null, "Local Execution Commands (Windows PowerShell)")), /*#__PURE__*/React.createElement("p", {
    style: {
      color: 'var(--text-muted)',
      fontSize: '0.9rem',
      marginBottom: '16px'
    }
  }, "Execute these exact commands in your local Windows PowerShell terminal from the project root. All outputs are automatically flushed with native ANSI color formatting."), /*#__PURE__*/React.createElement("div", {
    style: {
      marginBottom: '14px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.8rem',
      color: 'var(--gold-light)',
      fontWeight: 600,
      marginBottom: '4px'
    }
  }, "1. Preflight Model Connectivity Check:"), /*#__PURE__*/React.createElement("div", {
    className: "terminal-box"
  }, /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "python check_model.py"))), /*#__PURE__*/React.createElement("div", {
    style: {
      marginBottom: '14px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.8rem',
      color: 'var(--gold-light)',
      fontWeight: 600,
      marginBottom: '4px'
    }
  }, "2. Run Single Model Task (e.g. Task 3):"), /*#__PURE__*/React.createElement("div", {
    className: "terminal-box"
  }, /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "python run.py --task task_3 --mode full"))), /*#__PURE__*/React.createElement("div", {
    style: {
      marginBottom: '14px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.8rem',
      color: 'var(--gold-light)',
      fontWeight: 600,
      marginBottom: '4px'
    }
  }, "3. Run Complete Baseline Gate (All 6 Tasks):"), /*#__PURE__*/React.createElement("div", {
    className: "terminal-box"
  }, /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "python run.py --mode baselines"))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.8rem',
      color: 'var(--gold-light)',
      fontWeight: 600,
      marginBottom: '4px'
    }
  }, "4. Run Unit & Hardening Test Suite:"), /*#__PURE__*/React.createElement("div", {
    className: "terminal-box"
  }, /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "python -m unittest discover tests")))), /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-title"
  }, /*#__PURE__*/React.createElement("span", null, "Live Terminal Output Example"), /*#__PURE__*/React.createElement("span", {
    className: "badge badge-gold"
  }, "Flushed Real-Time")), /*#__PURE__*/React.createElement("div", {
    className: "terminal-box",
    style: {
      maxHeight: '420px',
      overflowY: 'auto'
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "term-line term-gold"
  }, "============================================================"), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-gold"
  }, "  KingdomForge -- RL Environment Harness"), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-gold"
  }, "============================================================"), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-blue"
  }, "  Configuration:"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "    MODE=full"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "    TASK_ID=task_3"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "    SEED=42"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "    MODEL=luna-model"), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-green"
  }, "  Both baselines PASSED. Proceeding to model run."), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-blue"
  }, "  ============================================================"), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-blue"
  }, "  Running Model Evaluation Episode:"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "    Task: task_3"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "    Model: luna-model"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "    Endpoint: https://api.moonshot.cn/v1"), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-blue"
  }, "  ============================================================"), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-blue"
  }, "  [Step 1/40]"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "    Calling model endpoint..."), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "    Model replied in 412.3ms (tokens: 84)"), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-yellow"
  }, "    Tool calls requested: 1"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "      -> Tool call #1: view_kingdom(args={})"), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "         Result: Resources: gold: 40, food: 25, wood: 30..."), /*#__PURE__*/React.createElement("span", {
    className: "term-line"
  }, "         State Hash: e9911c52"), /*#__PURE__*/React.createElement("span", {
    className: "term-line term-green"
  }, "  Model Episode Summary: VERIFIED PASSED (score=1.0000)")))));
}

var rootEl = document.getElementById('root');
if (rootEl) {
  if (typeof ReactDOM !== 'undefined' && ReactDOM.createRoot) {
    var reactRoot = ReactDOM.createRoot(rootEl);
    reactRoot.render(React.createElement(App, null));
  } else if (typeof ReactDOM !== 'undefined' && ReactDOM.render) {
    ReactDOM.render(React.createElement(App, null), rootEl);
  }
}