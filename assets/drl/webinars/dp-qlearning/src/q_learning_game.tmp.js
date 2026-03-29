
    async function init() {
      const response = await fetch("q_learning_policies.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} while loading q_learning_policies.json`);
      }
      const policyData = await response.json();
    const config = {
      rows: 3,
      cols: 3,
      startState: 0,
      goalState: 8,
      actionOrder: policyData.action_order,
    };

    const arrowMap = {
      UP: "↑",
      RIGHT: "→",
      DOWN: "↓",
      LEFT: "←",
    };

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
    const stateScores = policyData.final.state_scores;
    const qValues = policyData.final.Q;
    const transitionModel = policyData.final.transition_model;

    let state = config.startState;
    let totalReward = 0;
    let steps = 0;
    let terminal = false;
    let timer = null;

    function sampleIndex(weights) {
      const target = Math.random();
      let cumulative = 0;
      for (let i = 0; i < weights.length; i += 1) {
        cumulative += weights[i];
        if (target <= cumulative + Number.EPSILON) {
          return i;
        }
      }
      return weights.length - 1;
    }

    function choosePolicyAction(stateId) {
      return config.actionOrder[sampleIndex(finalPolicy[stateId])];
    }

    function chooseEnvironmentOutcome(stateId, actionName) {
      const outcomes = transitionModel[String(stateId)][actionName];
      return outcomes[sampleIndex(outcomes.map((outcome) => outcome.probability))];
    }

    function bestActionsForState(stateId) {
      const row = finalPolicy[stateId];
      const maxProb = Math.max(...row);
      return row
        .map((prob, index) => [prob, index])
        .filter(([prob]) => Math.abs(prob - maxProb) < 1e-12)
        .map(([, index]) => config.actionOrder[index]);
    }

    function renderBoard() {
      board.innerHTML = "";
      const minV = Math.min(...stateScores);
      const maxV = Math.max(...stateScores);
      const scale = maxV - minV || 1;

      for (let stateId = 0; stateId < finalPolicy.length; stateId += 1) {
        const cell = document.createElement("article");
        cell.className = "cell";
        if (stateId === state) {
          cell.classList.add("active");
        }
        if (stateId === config.goalState) {
          cell.classList.add("goal");
        }

        const normalized = (stateScores[stateId] - minV) / scale;
        cell.style.background = stateId === config.goalState
          ? ""
          : `linear-gradient(180deg, rgba(43, 108, 176, ${0.14 + normalized * 0.32}), rgba(255, 255, 255, 0.95))`;

        const bestActions = bestActionsForState(stateId)
          .map((action) => arrowMap[action])
          .join("");

        const lines = finalPolicy[stateId]
          .map((prob, index) => {
            const actionName = config.actionOrder[index];
            return `
              <div class="policy-line">
                <span>${arrowMap[actionName]}</span>
                <div class="bar"><span style="width:${(prob * 100).toFixed(1)}%"></span></div>
                <span>${prob.toFixed(3)}</span>
              </div>
            `;
          })
          .join("");

        cell.innerHTML = `
          <a class="cell-link" href="q_learning_model.html#state=${stateId}">
          <div class="cell-header">
            <span class="state-name">state ${stateId}</span>
            <div class="agent">🤖</div>
            <span class="state-value">${stateScores[stateId].toFixed(2)}</span>
          </div>
          <div class="state-name">best: ${bestActions || "G"}</div>
          <div class="policy-lines">${lines}</div>
          </a>
        `;
        board.appendChild(cell);
      }
    }

    function renderSummary(lastEvent = null) {
      modelLinkEl.href = `q_learning_model.html#state=${state}`;
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
          .map((actionName) => {
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
        })
        .join("");

      stateDetailEl.innerHTML = `
        <div class="stat" style="margin-bottom:6px;">
          <div class="stat-label">State Score max_a Q(s,a)</div>
          <div class="stat-value" style="color: var(--accent-deep);">${stateScores[state].toFixed(3)}</div>
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
          <div class="stat-label">Learned Q Row</div>
          <div style="margin-top:4px; line-height:1.25; font-size:0.76rem;">${qRow}</div>
        </div>
      `;
      transitionDetailEl.innerHTML = transitionRows;

      if (lastEvent) {
        const item = document.createElement("li");
        item.textContent = lastEvent;
        logEl.prepend(item);
      }
    }

    function resetGame() {
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
    }

    function stepGame() {
      if (terminal) {
        renderSummary("Episode already ended at the goal.");
        return;
      }

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

      if (terminal) {
        stepBtn.disabled = true;
        autoBtn.disabled = true;
        stopAuto();
      }
    }

    function autoplay(remaining = 8) {
      if (remaining <= 0 || terminal) {
        stopAuto();
        return;
      }
      stepGame();
      if (!terminal) {
        timer = window.setTimeout(() => autoplay(remaining - 1), 650);
      }
    }

    function stopAuto() {
      if (timer !== null) {
        window.clearTimeout(timer);
        timer = null;
      }
      stopBtn.disabled = true;
      if (!terminal) {
        stepBtn.disabled = false;
        autoBtn.disabled = false;
      }
    }

    resetBtn.addEventListener("click", resetGame);
    stepBtn.addEventListener("click", stepGame);
    autoBtn.addEventListener("click", () => {
      stopAuto();
      stepBtn.disabled = true;
      autoBtn.disabled = true;
      stopBtn.disabled = false;
      autoplay(8);
    });
    stopBtn.addEventListener("click", stopAuto);

    resetGame();
    }

    init().catch((error) => {
      console.error("Failed to load policy data", error);
      document.body.innerHTML = `<pre style="padding:16px;font-family:monospace;">Failed to load q_learning_policies.json\n${String(error)}</pre>`;
    });
  
