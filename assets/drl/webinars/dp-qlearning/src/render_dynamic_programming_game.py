"""Generate DP HTML views that fetch policy data at runtime."""

from __future__ import annotations

from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dynamic Programming Gridworld Game</title>
  <style>
    :root {{
      --bg: #f3efe5;
      --paper: #fffaf0;
      --ink: #1f2933;
      --muted: #5d6b78;
      --accent: #9b87d8;
      --accent-2: #c7b9ee;
      --accent-deep: #7058b2;
      --accent-soft: #f1ebff;
      --goal: #93c47d;
      --agent: rgba(59, 130, 246, 0.78);
      --agent-ring: rgba(59, 130, 246, 0.18);
      --grid: #d8cfc2;
      --shadow: 0 18px 40px rgba(31, 41, 51, 0.12);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: "Google Sans", "Product Sans", Inter, "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(224, 122, 95, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(43, 108, 176, 0.16), transparent 28%),
        linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
    }}

    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      min-height: 100vh;
      padding: 8px 12px 12px;
      display: grid;
      align-content: start;
    }}

    .hero {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      min-height: 96px;
      padding: 18px 18px 14px;
      border: 1px solid rgba(155, 135, 216, 0.22);
      border-radius: 20px;
      background:
        radial-gradient(circle at left top, rgba(199, 185, 238, 0.55), transparent 34%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.82), rgba(241, 235, 255, 0.92));
      box-shadow: 0 12px 26px rgba(112, 88, 178, 0.08);
    }}

    h1 {{
      margin: 0;
      font-size: clamp(1.45rem, 2vw, 2rem);
      line-height: 1.05;
      letter-spacing: -0.02em;
      font-weight: 700;
      color: var(--accent-deep);
      transform: translateY(-8px);
    }}

    .nav {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 0;
    }}

    .nav a {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      padding: 0 14px;
      border-radius: 999px;
      text-decoration: none;
      color: #fff;
      background: linear-gradient(135deg, var(--accent-deep), var(--accent));
      font-size: 0.88rem;
      border: 1px solid rgba(255, 255, 255, 0.24);
      box-shadow: 0 10px 20px rgba(112, 88, 178, 0.18);
    }}

    .nav a.secondary {{
      background: var(--accent-soft);
      color: var(--accent-deep);
      border-color: rgba(155, 135, 216, 0.28);
      box-shadow: none;
    }}

    .layout {{
      display: grid;
      gap: 12px;
      grid-template-columns: minmax(360px, 570px) minmax(820px, 1fr);
      align-items: stretch;
    }}

    .panel {{
      background: rgba(255, 250, 240, 0.9);
      border: 1px solid rgba(155, 135, 216, 0.14);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}

    .stage {{
      padding: 16px;
      height: 100%;
    }}

    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 10px;
    }}

    button {{
      appearance: none;
      border: 1px solid rgba(255, 255, 255, 0.24);
      border-radius: 999px;
      padding: 8px 14px;
      background: linear-gradient(135deg, var(--accent-deep), var(--accent));
      color: #fff;
      font: inherit;
      cursor: pointer;
      box-shadow: 0 10px 20px rgba(112, 88, 178, 0.18);
      transition: transform 120ms ease, opacity 120ms ease, box-shadow 120ms ease;
    }}

    button.secondary {{
      background: var(--accent-soft);
      color: var(--accent-deep);
      border-color: rgba(155, 135, 216, 0.28);
      box-shadow: none;
    }}

    button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 14px 26px rgba(112, 88, 178, 0.22);
    }}

    button:disabled {{
      opacity: 0.5;
      cursor: default;
      transform: none;
      box-shadow: none;
    }}

    .board {{
      display: grid;
      grid-template-columns: repeat(3, minmax(92px, 1fr));
      gap: 10px;
    }}

    .cell {{
      position: relative;
      min-height: 158px;
      border-radius: 20px;
      padding: 12px 12px 10px;
      border: 1px solid rgba(31, 41, 51, 0.1);
      background: rgba(255, 255, 255, 0.86);
      overflow: hidden;
    }}

    .cell-link {{
      color: inherit;
      text-decoration: none;
      display: block;
      height: 100%;
    }}

    .cell.goal {{
      border-color: rgba(104, 140, 84, 0.45);
      background: linear-gradient(180deg, rgba(147, 196, 125, 0.4), rgba(255, 250, 240, 0.95));
    }}

    .cell-header {{
      display: flex;
      justify-content: flex-start;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 6px;
      position: relative;
      z-index: 2;
    }}

    .state-name {{
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}

    .state-value {{
      font-size: 0.98rem;
      font-weight: 700;
      color: var(--accent-deep);
    }}

    .agent {{
      position: static;
      min-width: 30px;
      height: 30px;
      padding: 0 7px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid rgba(59, 130, 246, 0.2);
      box-shadow: 0 8px 18px rgba(59, 130, 246, 0.18);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: #2563eb;
      font-size: 0.95rem;
      transform: scale(0.92);
      opacity: 0;
      transition: transform 160ms ease, opacity 160ms ease;
    }}

    .cell.active .agent {{
      transform: scale(1);
      opacity: 1;
    }}

    .goal-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(104, 140, 84, 0.18);
      color: #355723;
      font-size: 0.68rem;
      font-weight: 700;
    }}

    .policy-lines {{
      position: relative;
      z-index: 2;
      display: grid;
      gap: 4px;
      margin-top: 8px;
    }}

    .policy-line {{
      display: grid;
      grid-template-columns: 16px 1fr 34px;
      gap: 6px;
      align-items: center;
      font-size: 0.74rem;
    }}

    .bar {{
      height: 8px;
      border-radius: 999px;
      background: rgba(31, 41, 51, 0.08);
      overflow: hidden;
    }}

    .bar > span {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}

    .summary {{
      padding: 16px;
      display: grid;
      gap: 12px;
      height: 100%;
    }}

    .summary-top {{
      display: grid;
      grid-template-columns: minmax(220px, 0.58fr) minmax(0, 1.42fr);
      gap: 12px;
      align-items: start;
    }}

    .summary-left {{
      display: grid;
      gap: 12px;
      align-content: start;
    }}

    .summary-right {{
      display: grid;
      grid-template-columns: minmax(0, 0.92fr) minmax(0, 1.28fr);
      gap: 12px;
      align-content: start;
    }}

    .summary-card {{
      padding: 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(155, 135, 216, 0.12);
    }}

    .summary-card h2 {{
      margin: 0 0 6px;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-deep);
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
    }}

    .stat {{
      padding: 6px 8px;
      border-radius: 10px;
      background: linear-gradient(180deg, #fcfaff, #f5f0ff);
    }}

    .stat-label {{
      color: var(--muted);
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .stat-value {{
      margin-top: 2px;
      font-size: 0.9rem;
      font-weight: 700;
    }}

    .log {{
      margin: 0;
      padding-left: 16px;
      max-height: 220px;
      overflow: auto;
      color: var(--muted);
      line-height: 1.28;
      font-size: 0.78rem;
    }}

    @media (max-width: 920px) {{
      .shell {{
        min-height: auto;
        align-content: start;
      }}

      .layout {{
        grid-template-columns: 1fr;
      }}

      .summary-top,
      .summary-right {{
        grid-template-columns: 1fr;
      }}

      .summary-left {{
        grid-template-columns: 1fr;
      }}

      .board {{
        grid-template-columns: repeat(3, minmax(80px, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Dynamic Programming Gridworld</h1>
      <div class="nav">
        <a id="model-link" href="dynamic_programming_model.html?state=0" class="secondary">Open Model Tables</a>
      </div>
    </section>

    <section class="layout">
      <div class="panel stage">
        <div class="controls">
          <button id="reset">Reset</button>
          <button id="step">Step Policy</button>
          <button id="auto">Autoplay 8 Steps</button>
          <button id="stop" class="secondary" disabled>Stop</button>
        </div>
        <div id="board" class="board"></div>
      </div>

      <aside class="panel summary">
        <div class="summary-top">
          <div class="summary-left">
            <div class="summary-card">
              <h2>Run State</h2>
              <div class="stats">
                <div class="stat">
                  <div class="stat-label">Current State</div>
                  <div class="stat-value" id="current-state">0</div>
                </div>
                <div class="stat">
                  <div class="stat-label">Total Reward</div>
                  <div class="stat-value" id="total-reward">0.0</div>
                </div>
                <div class="stat">
                  <div class="stat-label">Steps</div>
                  <div class="stat-value" id="steps">0</div>
                </div>
                <div class="stat">
                  <div class="stat-label">Terminal</div>
                  <div class="stat-value" id="terminal">No</div>
                </div>
              </div>
            </div>

            <div class="summary-card">
              <h2>Event Log</h2>
              <ol id="log" class="log"></ol>
            </div>
          </div>

          <div class="summary-right">
            <div class="summary-card">
              <h2>State Detail</h2>
              <div id="state-detail"></div>
            </div>

            <div class="summary-card">
              <h2>Transition Model</h2>
              <div id="transition-detail"></div>
            </div>
          </div>
        </div>
      </aside>
    </section>
  </div>

  <script>
    const policyData = __POLICY_JSON__;
    const config = {{
      rows: 3,
      cols: 3,
      startState: 0,
      goalState: 8,
      actionOrder: policyData.action_order,
    }};

    const arrowMap = {{
      UP: "↑",
      RIGHT: "→",
      DOWN: "↓",
      LEFT: "←",
    }};

    const board = document.getElementById("board");
    const currentStateEl = document.getElementById("current-state");
    const totalRewardEl = document.getElementById("total-reward");
    const stepsEl = document.getElementById("steps");
    const terminalEl = document.getElementById("terminal");
    const logEl = document.getElementById("log");
    const stateDetailEl = document.getElementById("state-detail");
    const transitionDetailEl = document.getElementById("transition-detail");
    const resetBtn = document.getElementById("reset");
    const stepBtn = document.getElementById("step");
    const autoBtn = document.getElementById("auto");
    const stopBtn = document.getElementById("stop");
    const modelLinkEl = document.getElementById("model-link");

    const finalPolicy = policyData.final.policy_matrix;
    const stateValues = policyData.final.V;
    const qValues = policyData.final.Q;
    const transitionModel = policyData.final.transition_model;

    let state = config.startState;
    let totalReward = 0;
    let steps = 0;
    let terminal = false;
    let timer = null;

    function sampleIndex(weights) {{
      const target = Math.random();
      let cumulative = 0;
      for (let i = 0; i < weights.length; i += 1) {{
        cumulative += weights[i];
        if (target <= cumulative + Number.EPSILON) {{
          return i;
        }}
      }}
      return weights.length - 1;
    }}

    function choosePolicyAction(stateId) {{
      return config.actionOrder[sampleIndex(finalPolicy[stateId])];
    }}

    function chooseEnvironmentOutcome(stateId, actionName) {{
      const outcomes = transitionModel[String(stateId)][actionName];
      return outcomes[sampleIndex(outcomes.map((outcome) => outcome.probability))];
    }}

    function bestActionsForState(stateId) {{
      const row = finalPolicy[stateId];
      const maxProb = Math.max(...row);
      return row
        .map((prob, index) => [prob, index])
        .filter(([prob]) => Math.abs(prob - maxProb) < 1e-12)
        .map(([, index]) => config.actionOrder[index]);
    }}

    function renderBoard() {{
      board.innerHTML = "";
      const minV = Math.min(...stateValues);
      const maxV = Math.max(...stateValues);
      const scale = maxV - minV || 1;

      for (let stateId = 0; stateId < finalPolicy.length; stateId += 1) {{
        const cell = document.createElement("article");
        cell.className = "cell";
        if (stateId === state) {{
          cell.classList.add("active");
        }}
        if (stateId === config.goalState) {{
          cell.classList.add("goal");
        }}

        const normalized = (stateValues[stateId] - minV) / scale;
        cell.style.background = stateId === config.goalState
          ? ""
          : `linear-gradient(180deg, rgba(43, 108, 176, ${0.14 + normalized * 0.32}), rgba(255, 255, 255, 0.95))`;

        const bestActions = bestActionsForState(stateId)
          .map((action) => arrowMap[action])
          .join("");

        const lines = finalPolicy[stateId]
          .map((prob, index) => {{
            const actionName = config.actionOrder[index];
            return `
              <div class="policy-line">
                <span>${arrowMap[actionName]}</span>
                <div class="bar"><span style="width:${(prob * 100).toFixed(1)}%"></span></div>
                <span>${prob.toFixed(3)}</span>
              </div>
            `;
          }})
          .join("");

        cell.innerHTML = `
          <a class="cell-link" href="dynamic_programming_model.html?state=${stateId}">
          <div class="cell-header">
            <span class="state-name">state ${stateId}</span>
            <div class="agent">🤖</div>
            <span class="state-value">${stateValues[stateId].toFixed(2)}</span>
          </div>
          <div class="state-name">best: ${bestActions || "G"}</div>
          <div class="policy-lines">${lines}</div>
          </a>
        `;
        board.appendChild(cell);
      }}
    }}

    function renderSummary(lastEvent = null) {{
      modelLinkEl.href = `dynamic_programming_model.html?state=${state}`;
      currentStateEl.textContent = String(state);
      totalRewardEl.textContent = totalReward.toFixed(1);
      stepsEl.textContent = String(steps);
      terminalEl.textContent = terminal ? "Yes" : "No";

      const bestActions = bestActionsForState(state);
      const qRow = qValues[state]
        .map((value, index) => `${config.actionOrder[index]}: ${value.toFixed(3)}`)
        .join("<br>");
      const policyRow = finalPolicy[state]
        .map((value, index) => `${config.actionOrder[index]}: ${value.toFixed(3)}`)
        .join("<br>");
        const transitionRows = config.actionOrder
          .map((actionName) => {{
            const outcomes = transitionModel[String(state)][actionName]
              .map((outcome) =>
                `p=${outcome.probability.toFixed(3)} → s${outcome.next_state}, r=${outcome.reward.toFixed(1)}, done=${outcome.done}`
              )
              .join("<br>");
            return `
            <div class="stat" style="margin-bottom:6px;">
              <div class="stat-label">${actionName}</div>
              <div style="margin-top:4px; line-height:1.25; font-size:0.76rem;">${outcomes}</div>
            </div>
          `;
        }})
        .join("");

      stateDetailEl.innerHTML = `
        <div class="stat" style="margin-bottom:6px;">
          <div class="stat-label">Value V*</div>
          <div class="stat-value" style="color: var(--accent-deep);">${stateValues[state].toFixed(3)}</div>
          ${state === config.goalState ? '<div style="margin-top:4px; font-size:0.72rem; font-weight:700; color:#355723;">GOAL</div>' : ''}
        </div>
        <div class="stat" style="margin-bottom:6px;">
          <div class="stat-label">Highest-probability actions</div>
          <div class="stat-value" style="font-size:0.86rem;">${bestActions.join(", ") || "GOAL"}</div>
        </div>
        <div class="stat" style="margin-bottom:6px;">
          <div class="stat-label">Policy Probabilities</div>
          <div style="margin-top:4px; line-height:1.25; font-size:0.76rem;">${policyRow}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Q* Row</div>
          <div style="margin-top:4px; line-height:1.25; font-size:0.76rem;">${qRow}</div>
        </div>
      `;
      transitionDetailEl.innerHTML = transitionRows;

      if (lastEvent) {{
        const item = document.createElement("li");
        item.textContent = lastEvent;
        logEl.prepend(item);
      }}
    }}

    function resetGame() {{
      stopAuto();
      state = config.startState;
      totalReward = 0;
      steps = 0;
      terminal = false;
      logEl.innerHTML = "";
      renderBoard();
      renderSummary("Reset to state 0.");
      stepBtn.disabled = false;
      autoBtn.disabled = false;
    }}

    function stepGame() {{
      if (terminal) {{
        renderSummary("Episode already ended at the goal.");
        return;
      }}

      const policyAction = choosePolicyAction(state);
      const outcome = chooseEnvironmentOutcome(state, policyAction);
      const nextState = outcome.next_state;
      const reward = outcome.reward;
      const reachedGoal = outcome.done;

      const previousState = state;
      state = nextState;
      totalReward += reward;
      steps += 1;
      terminal = reachedGoal;

      renderBoard();
      renderSummary(
        `s${previousState} -> s${nextState}: policy chose ${policyAction}, sampled p=${outcome.probability.toFixed(3)}, reward ${reward.toFixed(1)}`
      );

      if (terminal) {{
        stepBtn.disabled = true;
        autoBtn.disabled = true;
        stopAuto();
      }}
    }}

    function autoplay(remaining = 8) {{
      if (remaining <= 0 || terminal) {{
        stopAuto();
        return;
      }}
      stepGame();
      if (!terminal) {{
        timer = window.setTimeout(() => autoplay(remaining - 1), 650);
      }}
    }}

    function stopAuto() {{
      if (timer !== null) {{
        window.clearTimeout(timer);
        timer = null;
      }}
      stopBtn.disabled = true;
      if (!terminal) {{
        stepBtn.disabled = false;
        autoBtn.disabled = false;
      }}
    }}

    resetBtn.addEventListener("click", resetGame);
    stepBtn.addEventListener("click", stepGame);
    autoBtn.addEventListener("click", () => {{
      stopAuto();
      stepBtn.disabled = true;
      autoBtn.disabled = true;
      stopBtn.disabled = false;
      autoplay(8);
    }});
    stopBtn.addEventListener("click", stopAuto);

    resetGame();
  </script>
</body>
</html>
"""


MODEL_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dynamic Programming Model Tables</title>
  <style>
    :root {
      --bg: #f3efe5;
      --paper: #fffaf0;
      --ink: #1f2933;
      --muted: #5d6b78;
      --accent: #9b87d8;
      --accent-2: #c7b9ee;
      --accent-deep: #7058b2;
      --accent-soft: #f1ebff;
      --line: #e5daf7;
      --shadow: 0 18px 40px rgba(31, 41, 51, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      font-family: "Google Sans", "Product Sans", Inter, "Segoe UI", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(199, 185, 238, 0.26), transparent 28%),
        radial-gradient(circle at top right, rgba(155, 135, 216, 0.14), transparent 24%),
        linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
    }

    .shell {
      max-width: 1380px;
      margin: 0 auto;
      padding: 16px 12px 24px;
    }

    .hero {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
      min-height: 96px;
      padding: 18px 18px 14px;
      border: 1px solid rgba(155, 135, 216, 0.22);
      border-radius: 20px;
      background:
        radial-gradient(circle at left top, rgba(199, 185, 238, 0.55), transparent 34%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.82), rgba(241, 235, 255, 0.92));
      box-shadow: 0 12px 26px rgba(112, 88, 178, 0.08);
    }

    h1 {
      margin: 0;
      font-size: clamp(1.35rem, 1.9vw, 1.85rem);
      line-height: 1.05;
      letter-spacing: -0.02em;
      font-weight: 700;
      color: var(--accent-deep);
    }

    .nav {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .nav a, .picker button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 34px;
      padding: 0 12px;
      border-radius: 999px;
      text-decoration: none;
      color: #fff;
      background: linear-gradient(135deg, var(--accent-deep), var(--accent));
      border: 1px solid rgba(255, 255, 255, 0.24);
      cursor: pointer;
      font: inherit;
      box-shadow: 0 10px 20px rgba(112, 88, 178, 0.18);
      white-space: nowrap;
    }

    select {
      width: 100%;
      min-height: 42px;
      border-radius: 14px;
      border: 1px solid rgba(155, 135, 216, 0.2);
      background: rgba(255, 255, 255, 0.92);
      color: var(--ink);
      padding: 0 12px;
      font: inherit;
    }

    .nav a.secondary, .picker button.secondary {
      background: var(--accent-soft);
      color: var(--accent-deep);
      border-color: rgba(155, 135, 216, 0.28);
      box-shadow: none;
    }

    .layout {
      display: grid;
      gap: 12px;
      grid-template-columns: minmax(260px, 320px) minmax(0, 1fr);
      align-items: start;
    }

    .panel {
      background: rgba(255, 250, 240, 0.9);
      border: 1px solid rgba(155, 135, 216, 0.14);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 14px;
    }

    .stack {
      display: grid;
      gap: 12px;
    }

    .panel h2 {
      margin: 0 0 8px;
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-deep);
    }

    .panel-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
    }

    .picker {
      display: grid;
      gap: 10px;
    }

    .picker-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }

    .picker-grid button.active {
      background: linear-gradient(135deg, var(--accent-deep), var(--accent));
    }

    .meta {
      display: grid;
      gap: 10px;
    }

    .meta-item {
      padding: 12px;
      border-radius: 16px;
      background: linear-gradient(180deg, #fcfaff, #f5f0ff);
    }

    .meta-label {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .meta-value {
      margin-top: 5px;
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--accent-deep);
    }

    .tables {
      display: grid;
      gap: 10px;
    }

    .matrix-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      background: rgba(255, 255, 255, 0.82);
      border-radius: 16px;
      overflow: hidden;
    }

    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.86rem;
    }

    th {
      background: linear-gradient(180deg, #f7f1ff, #efe6ff);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }

    tr:last-child td {
      border-bottom: 0;
    }

    td.tie {
      background: rgba(155, 135, 216, 0.16);
      font-weight: 700;
    }

    .caption {
      margin: 0;
      font-size: 0.84rem;
      color: var(--muted);
      white-space: nowrap;
    }

    .grid-table td:first-child,
    .grid-table th:first-child {
      width: 72px;
    }

    .small {
      font-size: 0.82rem;
      color: var(--muted);
    }

    .transition-table th,
    .transition-table td {
      padding: 4px 6px;
      font-size: 0.72rem;
    }

    @media (max-width: 920px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .matrix-row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Dynamic Programming Model Tables</h1>
      <div class="nav">
        <a href="dynamic_programming_game.html">Open Game View</a>
      </div>
    </section>

    <section class="layout">
      <aside class="stack">
        <div class="panel">
          <h2>State Picker</h2>
          <div class="picker">
            <select id="state-select"></select>
            <div id="state-picker" class="picker-grid"></div>
          </div>
        </div>

        <div class="panel">
          <h2>Current State</h2>
          <div class="meta">
            <div class="meta-item">
              <div class="meta-label">State</div>
              <div class="meta-value" id="current-state">0</div>
            </div>
            <div class="meta-item">
              <div class="meta-label">V*</div>
              <div class="meta-value" id="current-v">0.000</div>
            </div>
            <div class="meta-item">
              <div class="meta-label">Highest Policy Actions</div>
              <div class="meta-value" id="current-best">RIGHT, DOWN</div>
            </div>
          </div>
        </div>

        <div class="panel">
          <h2>Export Summary</h2>
          <div class="small">
            <div>Epsilon: <span id="epsilon"></span></div>
            <div>Stable: <span id="stable"></span></div>
            <div>Action Order: <span id="action-order"></span></div>
          </div>
        </div>
      </aside>

      <main class="tables">
        <div class="matrix-row">
          <div class="panel">
            <div class="panel-head">
              <h2>Policy Matrix</h2>
              <p class="caption">Rows are states. Columns follow the exported action order.</p>
            </div>
            <div id="policy-table"></div>
          </div>

          <div class="panel">
            <div class="panel-head">
              <h2>Q Matrix</h2>
              <p class="caption">Tied maxima are highlighted in each row.</p>
            </div>
            <div id="q-table"></div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <h2>Transition Model For Current State</h2>
            <p class="caption">Flat table view of the exported transition model <code>P[s][a]</code>.</p>
          </div>
          <div id="transition-table"></div>
        </div>
      </main>
    </section>
  </div>

  <script>
    const policyData = __POLICY_JSON__;
    const actionOrder = policyData.action_order;
    const finalData = policyData.final;
    const transitionModel = finalData.transition_model;
    const policyMatrix = finalData.policy_matrix;
    const qMatrix = finalData.Q;
    const vValues = finalData.V;
    let selectedState = 0;

    const pickerEl = document.getElementById("state-picker");
    const stateSelectEl = document.getElementById("state-select");
    const currentStateEl = document.getElementById("current-state");
    const currentVEl = document.getElementById("current-v");
    const currentBestEl = document.getElementById("current-best");
    const epsilonEl = document.getElementById("epsilon");
    const stableEl = document.getElementById("stable");
    const actionOrderEl = document.getElementById("action-order");
    const policyTableEl = document.getElementById("policy-table");
    const qTableEl = document.getElementById("q-table");
    const transitionTableEl = document.getElementById("transition-table");

    function bestIndices(row) {
      const maxValue = Math.max(...row);
      return row
        .map((prob, index) => [prob, index])
        .filter(([prob]) => Math.abs(prob - maxValue) < 1e-12)
        .map(([, index]) => index);
    }

    function bestActionsForState(stateId) {
      return bestIndices(policyMatrix[stateId]).map((index) => actionOrder[index]);
    }

    function buildMatrixTable(title, matrix, digits = 3) {
      const header = actionOrder.map((action) => `<th>${action}</th>`).join("");
      const rows = matrix.map((row, state) => {
        const tied = new Set(bestIndices(row));
        const cols = row.map((value, index) => `<td class="${tied.has(index) ? "tie" : ""}">${Number(value).toFixed(digits)}</td>`).join("");
        return `<tr><td><strong>${state}</strong></td>${cols}</tr>`;
      }).join("");
      return `
        <table class="grid-table">
          <thead>
            <tr><th>State</th>${header}</tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }

    function renderPicker() {
      pickerEl.innerHTML = "";
      stateSelectEl.innerHTML = "";
      for (let state = 0; state < policyMatrix.length; state += 1) {
        const option = document.createElement("option");
        option.value = String(state);
        option.textContent = `State ${state}`;
        option.selected = state === selectedState;
        stateSelectEl.appendChild(option);

        const button = document.createElement("button");
        button.textContent = `s${state}`;
        button.className = state === selectedState ? "active" : "secondary";
        button.addEventListener("click", () => {
          selectedState = state;
          render();
        });
        pickerEl.appendChild(button);
      }
    }

    function renderTransitionTable() {
      const rows = [];
      for (const actionName of actionOrder) {
        const outcomes = transitionModel[String(selectedState)][actionName];
        for (const outcome of outcomes) {
          rows.push(`
            <tr>
              <td>${selectedState}</td>
              <td>${actionName}</td>
              <td>${Number(outcome.probability).toFixed(3)}</td>
              <td>${outcome.next_state}</td>
              <td>${Number(outcome.reward).toFixed(1)}</td>
              <td>${outcome.done}</td>
            </tr>
          `);
        }
      }

      transitionTableEl.innerHTML = `
        <table class="transition-table">
          <thead>
            <tr>
              <th>State</th>
              <th>Action</th>
              <th>Probability</th>
              <th>Next State</th>
              <th>Reward</th>
              <th>Done</th>
            </tr>
          </thead>
          <tbody>${rows.join("")}</tbody>
        </table>
      `;
    }

    function render() {
      const url = new URL(window.location.href);
      url.searchParams.set("state", String(selectedState));
      window.history.replaceState({}, "", url);
      renderPicker();
      currentStateEl.textContent = String(selectedState);
      currentVEl.textContent = Number(vValues[selectedState]).toFixed(3);
      currentBestEl.textContent = bestActionsForState(selectedState).join(", ");
      epsilonEl.textContent = String(policyData.epsilon);
      stableEl.textContent = String(finalData.stable);
      actionOrderEl.textContent = actionOrder.join(", ");
      policyTableEl.innerHTML = buildMatrixTable("Policy", policyMatrix, 3);
      qTableEl.innerHTML = buildMatrixTable("Q", qMatrix, 3);
      renderTransitionTable();
    }

    function initializeSelectedState() {
      const params = new URLSearchParams(window.location.search);
      const rawState = Number.parseInt(params.get("state") ?? "0", 10);
      if (Number.isInteger(rawState) && rawState >= 0 && rawState < policyMatrix.length) {
        selectedState = rawState;
      }
    }

    stateSelectEl.addEventListener("change", (event) => {
      selectedState = Number.parseInt(event.target.value, 10);
      render();
    });

    initializeSelectedState();
    render();
  </script>
</body>
</html>
"""


def convert_to_fetch_template(template: str, json_filename: str) -> str:
    """Wrap an inline-data HTML template so it fetches JSON at runtime.

    Precondition:
    `template` contains the expected placeholder script structure used by the
    case-study HTML templates, and `json_filename` names the exported policy
    data file that should be fetched by the browser.

    What happens:
    1. Replace the inline `policyData` bootstrap with an async `fetch()` call.
    2. Append common error handling that renders a readable failure message.
    3. Preserve the existing template-specific initialization calls.

    Postcondition:
    Returns HTML that loads the policy JSON dynamically instead of embedding it
    directly in the page source.
    """
    script_start = f"""<script>
    async function init() {{
      const response = await fetch("{json_filename}", {{ cache: "no-store" }});
      if (!response.ok) {{
        throw new Error(`HTTP ${{response.status}} while loading {json_filename}`);
      }}
      const policyData = await response.json();
"""
    script_end = f"""
    }}

    init().catch((error) => {{
      console.error("Failed to load policy data", error);
      document.body.innerHTML = `<pre style="padding:16px;font-family:monospace;">Failed to load {json_filename}\\n${{String(error)}}</pre>`;
    }});
  </script>"""
    template = template.replace('<script>\n    const policyData = __POLICY_JSON__;\n', script_start)
    template = template.replace("\n    resetGame();\n  </script>", "\n    resetGame();" + script_end)
    template = template.replace("\n    initializeSelectedState();\n    render();\n  </script>", "\n    initializeSelectedState();\n    render();" + script_end)
    return template


def main() -> None:
    """Generate the dynamic-programming game and model HTML files.

    Precondition:
    `dynamic_programming_policies.json` already exists in the same directory.

    What happens:
    1. Resolve the working paths for the JSON export and output HTML files.
    2. Fail fast if the JSON export is missing.
    3. Convert both base templates into fetch-based HTML documents.
    4. Normalize escaped braces introduced by Python formatting.
    5. Write the final game and model pages to disk.

    Postcondition:
    `dynamic_programming_game.html` and `dynamic_programming_model.html` are
    regenerated from the latest exported policy data.
    """
    workdir = Path(__file__).resolve().parent
    json_path = workdir / "dynamic_programming_policies.json"
    game_html_path = workdir / "dynamic_programming_game.html"
    model_html_path = workdir / "dynamic_programming_model.html"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing precondition: generate {json_path.name} before rendering the dynamic programming HTML views."
        )

    html = convert_to_fetch_template(HTML_TEMPLATE, json_path.name)
    html = html.replace("{{", "{").replace("}}", "}")
    game_html_path.write_text(html, encoding="utf-8")

    model_html = convert_to_fetch_template(MODEL_TEMPLATE, json_path.name)
    model_html = model_html.replace("{{", "{").replace("}}", "}")
    model_html_path.write_text(model_html, encoding="utf-8")
    print(game_html_path.resolve())
    print(model_html_path.resolve())


if __name__ == "__main__":
    main()
