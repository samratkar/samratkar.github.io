
    async function init() {
      const response = await fetch("q_learning_policies.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} while loading q_learning_policies.json`);
      }
      const policyData = await response.json();
    const actionOrder = policyData.action_order;
    const finalData = policyData.final;
    const policyMatrix = finalData.policy_matrix;
    const qMatrix = finalData.Q;
    const stateScores = finalData.state_scores;
    let selectedState = 0;

    const pickerEl = document.getElementById("state-picker");
    const stateSelectEl = document.getElementById("state-select");
    const currentStateEl = document.getElementById("current-state");
    const currentScoreEl = document.getElementById("current-score");
    const currentBestEl = document.getElementById("current-best");
    const numEpisodesEl = document.getElementById("num-episodes");
    const alphaEl = document.getElementById("alpha");
    const gammaEl = document.getElementById("gamma");
    const epsilonStartEl = document.getElementById("epsilon-start");
    const epsilonFinalEl = document.getElementById("epsilon-final");
    const actionOrderEl = document.getElementById("action-order");
    const policyTableEl = document.getElementById("policy-table");
    const qTableEl = document.getElementById("q-table");

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

    function render() {
      window.history.replaceState({}, "", `#state=${selectedState}`);
      renderPicker();
      currentStateEl.textContent = String(selectedState);
      currentScoreEl.textContent = Number(stateScores[selectedState]).toFixed(3);
      currentBestEl.textContent = bestActionsForState(selectedState).join(", ");
      numEpisodesEl.textContent = String(policyData.num_episodes);
      alphaEl.textContent = String(policyData.alpha);
      gammaEl.textContent = String(policyData.gamma);
      epsilonStartEl.textContent = String(policyData.epsilon_start);
      epsilonFinalEl.textContent = String(policyData.epsilon_final);
      actionOrderEl.textContent = actionOrder.join(", ");
      policyTableEl.innerHTML = buildMatrixTable("Policy", policyMatrix, 3);
      qTableEl.innerHTML = buildMatrixTable("Q", qMatrix, 3);
    }

    function initializeSelectedState() {
      const hashMatch = window.location.hash.match(/(?:^#|&)state=(\d+)/);
      const searchState = new URLSearchParams(window.location.search).get("state");
      const rawState = Number.parseInt(hashMatch?.[1] ?? searchState ?? "0", 10);
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
    }

    init().catch((error) => {
      console.error("Failed to load policy data", error);
      document.body.innerHTML = `<pre style="padding:16px;font-family:monospace;">Failed to load q_learning_policies.json\n${String(error)}</pre>`;
    });
  
