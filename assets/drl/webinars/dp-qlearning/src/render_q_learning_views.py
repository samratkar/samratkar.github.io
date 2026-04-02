"""Generate Q-learning HTML views that fetch policy data at runtime."""

from __future__ import annotations

from pathlib import Path
import re

from render_dynamic_programming_game import HTML_TEMPLATE, MODEL_TEMPLATE


def convert_to_fetch_template(template: str, json_filename: str) -> str:
    """Wrap an inline-data HTML template so it fetches Q-learning JSON at runtime.

    Precondition:
    `template` follows the same placeholder conventions as the shared HTML
    templates, and `json_filename` is the name of the exported Q-learning JSON
    payload to load in the browser.

    What happens:
    1. Replace the inline `policyData` bootstrap with an async `fetch()` call.
    2. Inject shared error handling for failed JSON loads.
    3. Preserve the page-specific initialization logic already present in the
       template.

    Postcondition:
    Returns HTML that loads the JSON artifact dynamically rather than embedding
    it at generation time.
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


def build_game_template() -> str:
    """Specialize the shared game template for Q-learning terminology and links.

    Precondition:
    The imported `HTML_TEMPLATE` still contains the dynamic-programming labels
    and snippets targeted by the replacement list in this function.

    What happens:
    1. Start from the shared dynamic-programming game template.
    2. Replace titles, navigation targets, and displayed metric labels so the
       page reflects the Q-learning export schema.
    3. Swap value-function references for the Q-learning state-score fields.

    Postcondition:
    Returns an HTML template string ready to be wrapped with runtime JSON
    loading for the Q-learning game view.
    """
    template = HTML_TEMPLATE
    replacements = [
        ("Dynamic Programming Gridworld Game", "Q-Learning Gridworld Game"),
        ("Dynamic Programming Gridworld", "Q-Learning Gridworld"),
        ("dynamic_programming_model.html?state=0", "q_learning_model.html#state=0"),
        ('href="dynamic_programming_model.html?state=${stateId}"', 'href="q_learning_model.html#state=${stateId}"'),
        ("dynamic_programming_model.html?state=${state}", "q_learning_model.html#state=${state}"),
        ('\n            <div class="summary-card">\n              <h2>Transition Model</h2>\n              <div id="transition-detail"></div>\n            </div>', ""),
        ("const stateValues = policyData.final.V;", "const stateScores = policyData.final.state_scores;"),
        ("const qValues = policyData.final.Q;", "const qValues = policyData.final.Q;"),
        ("const transitionModel = policyData.final.transition_model;\n", ""),
        ('const transitionDetailEl = document.getElementById("transition-detail");\n', ""),
        ("Math.min(...stateValues)", "Math.min(...stateScores)"),
        ("Math.max(...stateValues)", "Math.max(...stateScores)"),
        ("(stateValues[stateId] - minV) / scale", "(stateScores[stateId] - minV) / scale"),
        ("${stateValues[stateId].toFixed(2)}", "${stateScores[stateId].toFixed(2)}"),
        ("Value V*", "State Score max_a Q(s,a)"),
        ("${stateValues[state].toFixed(3)}", "${stateScores[state].toFixed(3)}"),
        ('<div class="stat-label">Q* Row</div>', '<div class="stat-label">Learned Q Row</div>'),
        ('        const transitionRows = config.actionOrder\n          .map((actionName) => {\n            const outcomes = transitionModel[String(state)][actionName]\n              .map((outcome) =>\n                `p=${outcome.probability.toFixed(3)} → s${outcome.next_state}, r=${outcome.reward.toFixed(1)}, done=${outcome.done}`\n              )\n              .join("<br>");\n            return `\n            <div class="stat" style="margin-bottom:6px;">\n              <div class="stat-label">${actionName}</div>\n              <div style="margin-top:4px; line-height:1.25; font-size:0.76rem;">${outcomes}</div>\n            </div>\n          `;\n        })\n        .join("");\n', ""),
        ('        const transitionRows = config.actionOrder\n          .map((actionName) => {{\n            const outcomes = transitionModel[String(state)][actionName]\n              .map((outcome) =>\n                `p=${{outcome.probability.toFixed(3)}} → s${{outcome.next_state}}, r=${{outcome.reward.toFixed(1)}}, done=${{outcome.done}}`\n              )\n              .join("<br>");\n            return `\n            <div class="stat" style="margin-bottom:6px;">\n              <div class="stat-label">${{actionName}}</div>\n              <div style="margin-top:4px; line-height:1.25; font-size:0.76rem;">${{outcomes}}</div>\n            </div>\n          `;\n        }})\n        .join("");\n', ""),
        ("      transitionDetailEl.innerHTML = transitionRows;\n", ""),
        ('    function chooseEnvironmentOutcome(stateId, actionName) {\n      const outcomes = transitionModel[String(stateId)][actionName];\n      return outcomes[sampleIndex(outcomes.map((outcome) => outcome.probability))];\n    }\n', '    function moveState(stateId, actionName) {\n      if (stateId === config.goalState) {\n        return stateId;\n      }\n\n      const row = Math.floor(stateId / config.cols);\n      const col = stateId % config.cols;\n      let nextRow = row;\n      let nextCol = col;\n\n      if (actionName === "UP") {\n        nextRow = Math.max(0, row - 1);\n      } else if (actionName === "RIGHT") {\n        nextCol = Math.min(config.cols - 1, col + 1);\n      } else if (actionName === "DOWN") {\n        nextRow = Math.min(config.rows - 1, row + 1);\n      } else if (actionName === "LEFT") {\n        nextCol = Math.max(0, col - 1);\n      }\n\n      return nextRow * config.cols + nextCol;\n    }\n'),
        ('    function chooseEnvironmentOutcome(stateId, actionName) {{\n      const outcomes = transitionModel[String(stateId)][actionName];\n      return outcomes[sampleIndex(outcomes.map((outcome) => outcome.probability))];\n    }}\n', '    function moveState(stateId, actionName) {{\n      if (stateId === config.goalState) {{\n        return stateId;\n      }}\n\n      const row = Math.floor(stateId / config.cols);\n      const col = stateId % config.cols;\n      let nextRow = row;\n      let nextCol = col;\n\n      if (actionName === "UP") {{\n        nextRow = Math.max(0, row - 1);\n      }} else if (actionName === "RIGHT") {{\n        nextCol = Math.min(config.cols - 1, col + 1);\n      }} else if (actionName === "DOWN") {{\n        nextRow = Math.min(config.rows - 1, row + 1);\n      }} else if (actionName === "LEFT") {{\n        nextCol = Math.max(0, col - 1);\n      }}\n\n      return nextRow * config.cols + nextCol;\n    }}\n'),
        ('      const policyAction = choosePolicyAction(state);\n      const outcome = chooseEnvironmentOutcome(state, policyAction);\n      const nextState = outcome.next_state;\n      const reward = outcome.reward;\n      const reachedGoal = outcome.done;\n', '      const policyAction = choosePolicyAction(state);\n      const nextState = moveState(state, policyAction);\n      const reachedGoal = nextState === config.goalState;\n      const reward = reachedGoal ? 10 : -1;\n'),
        ("        `s${previousState} -> s${nextState}: policy chose ${policyAction}, sampled p=${outcome.probability.toFixed(3)}, reward ${reward.toFixed(1)}`\n", "        `s${previousState} -> s${nextState}: policy chose ${policyAction}, reward ${reward.toFixed(1)}`\n"),
    ]
    for old, new in replacements:
        template = template.replace(old, new)
    template = re.sub(
        r"\n\s*const transitionRows = config\.actionOrder.*?\.join\(\"\"\);\n",
        "\n",
        template,
        flags=re.DOTALL,
    )
    template = template.replace("      transitionDetailEl.innerHTML = transitionRows;\n", "")
    return template


def build_model_template() -> str:
    """Specialize the shared model-table template for Q-learning exports.

    Precondition:
    The imported `MODEL_TEMPLATE` still contains the expected dynamic-
    programming strings and DOM ids referenced by the replacement rules below.

    What happens:
    1. Start from the shared model-table template.
    2. Replace navigation, labels, metadata fields, and URL-state handling so
       they match the Q-learning view and its JSON structure.
    3. Update the rendered metrics from value-function terminology to the
       exported Q-learning scores and hyperparameters.

    Postcondition:
    Returns an HTML template string configured for the Q-learning model view.
    """
    template = MODEL_TEMPLATE
    replacements = [
        ("Dynamic Programming Model Tables", "Q-Learning Model Tables"),
        ('href="dynamic_programming_game.html"', 'href="q_learning_game.html"'),
        ('const url = new URL(window.location.href);\n      url.searchParams.set("state", String(selectedState));\n      window.history.replaceState({}, "", url);', 'window.history.replaceState({}, "", `#state=${selectedState}`);'),
        ('const params = new URLSearchParams(window.location.search);\n      const rawState = Number.parseInt(params.get("state") ?? "0", 10);', 'const hashMatch = window.location.hash.match(/(?:^#|&)state=(\\d+)/);\n      const searchState = new URLSearchParams(window.location.search).get("state");\n      const rawState = Number.parseInt(hashMatch?.[1] ?? searchState ?? "0", 10);'),
        ('<div class="meta-label">V*</div>', '<div class="meta-label">max_a Q(s,a)</div>'),
        ('id="current-v"', 'id="current-score"'),
        ("RIGHT, DOWN", "RIGHT"),
        ('<div>Epsilon: <span id="epsilon"></span></div>', '<div>Episodes: <span id="num-episodes"></span></div>'),
        ('<div>Stable: <span id="stable"></span></div>', '<div>Alpha: <span id="alpha"></span></div>'),
        ('<div>Action Order: <span id="action-order"></span></div>', '<div>Gamma: <span id="gamma"></span></div><div>Epsilon Start: <span id="epsilon-start"></span></div><div>Epsilon Final: <span id="epsilon-final"></span></div><div>Action Order: <span id="action-order"></span></div>'),
        ('\n        <div class="panel">\n          <div class="panel-head">\n            <h2>Transition Model For Current State</h2>\n            <p class="caption">Flat table view of the exported transition model <code>P[s][a]</code>.</p>\n          </div>\n          <div id="transition-table"></div>\n        </div>', ""),
        ("const vValues = finalData.V;", "const stateScores = finalData.state_scores;"),
        ("const transitionModel = finalData.transition_model;\n", ""),
        ('const currentVEl = document.getElementById("current-v");', 'const currentScoreEl = document.getElementById("current-score");'),
        ('const epsilonEl = document.getElementById("epsilon");', 'const numEpisodesEl = document.getElementById("num-episodes");'),
        ('const stableEl = document.getElementById("stable");', 'const alphaEl = document.getElementById("alpha");'),
        ('const actionOrderEl = document.getElementById("action-order");', 'const gammaEl = document.getElementById("gamma");\n    const epsilonStartEl = document.getElementById("epsilon-start");\n    const epsilonFinalEl = document.getElementById("epsilon-final");\n    const actionOrderEl = document.getElementById("action-order");'),
        ('const transitionTableEl = document.getElementById("transition-table");\n', ""),
        ('\n    function renderTransitionTable() {\n      const rows = [];\n      for (const actionName of actionOrder) {\n        const outcomes = transitionModel[String(selectedState)][actionName];\n        for (const outcome of outcomes) {\n          rows.push(`\n            <tr>\n              <td>${selectedState}</td>\n              <td>${actionName}</td>\n              <td>${Number(outcome.probability).toFixed(3)}</td>\n              <td>${outcome.next_state}</td>\n              <td>${Number(outcome.reward).toFixed(1)}</td>\n              <td>${outcome.done}</td>\n            </tr>\n          `);\n        }\n      }\n\n      transitionTableEl.innerHTML = `\n        <table class="transition-table">\n          <thead>\n            <tr>\n              <th>State</th>\n              <th>Action</th>\n              <th>Probability</th>\n              <th>Next State</th>\n              <th>Reward</th>\n              <th>Done</th>\n            </tr>\n          </thead>\n          <tbody>${rows.join("")}</tbody>\n        </table>\n      `;\n    }\n', ""),
        ("currentVEl.textContent = Number(vValues[selectedState]).toFixed(3);", "currentScoreEl.textContent = Number(stateScores[selectedState]).toFixed(3);"),
        ("epsilonEl.textContent = String(policyData.epsilon);", "numEpisodesEl.textContent = String(policyData.num_episodes);"),
        ("stableEl.textContent = String(finalData.stable);", "alphaEl.textContent = String(policyData.alpha);"),
        ('actionOrderEl.textContent = actionOrder.join(", ");', 'gammaEl.textContent = String(policyData.gamma);\n      epsilonStartEl.textContent = String(policyData.epsilon_start);\n      epsilonFinalEl.textContent = String(policyData.epsilon_final);\n      actionOrderEl.textContent = actionOrder.join(", ");'),
        ("      renderTransitionTable();\n", ""),
    ]
    for old, new in replacements:
        template = template.replace(old, new)
    return template


def main() -> None:
    """Generate the Q-learning game and model HTML files from the JSON export.

    Precondition:
    `q_learning_policies.json` already exists in the same directory.

    What happens:
    1. Resolve the local paths for the JSON artifact and output HTML files.
    2. Fail early if the JSON export has not been generated yet.
    3. Build the Q-learning-specific game and model templates.
    4. Wrap both templates so they fetch the JSON payload at runtime.
    5. Normalize escaped braces and write the final HTML files to disk.

    Postcondition:
    `q_learning_game.html` and `q_learning_model.html` are regenerated from the
    latest Q-learning export.
    """
    workdir = Path(__file__).resolve().parent
    json_path = workdir / "q_learning_policies.json"
    game_html_path = workdir / "q_learning_game.html"
    model_html_path = workdir / "q_learning_model.html"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing precondition: generate {json_path.name} before rendering the Q-learning HTML views."
        )

    game_html = convert_to_fetch_template(build_game_template(), json_path.name)
    game_html = game_html.replace("{{", "{").replace("}}", "}")
    game_html_path.write_text(game_html, encoding="utf-8")

    model_html = convert_to_fetch_template(build_model_template(), json_path.name)
    model_html = model_html.replace("{{", "{").replace("}}", "}")
    model_html_path.write_text(model_html, encoding="utf-8")

    print(game_html_path.resolve())
    print(model_html_path.resolve())


if __name__ == "__main__":
    main()
