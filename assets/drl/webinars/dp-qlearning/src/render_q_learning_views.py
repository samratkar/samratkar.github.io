"""Generate Q-learning HTML views that fetch policy data at runtime."""

from __future__ import annotations

from pathlib import Path

from render_dynamic_programming_game import HTML_TEMPLATE, MODEL_TEMPLATE


def convert_to_fetch_template(template: str, json_filename: str) -> str:
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
    template = HTML_TEMPLATE
    replacements = [
        ("Dynamic Programming Gridworld Game", "Q-Learning Gridworld Game"),
        ("Dynamic Programming Gridworld", "Q-Learning Gridworld"),
        ("dynamic_programming_model.html?state=0", "q_learning_model.html#state=0"),
        ('href="dynamic_programming_model.html?state=${stateId}"', 'href="q_learning_model.html#state=${stateId}"'),
        ("dynamic_programming_model.html?state=${state}", "q_learning_model.html#state=${state}"),
        ("const stateValues = policyData.final.V;", "const stateScores = policyData.final.state_scores;"),
        ("const qValues = policyData.final.Q;", "const qValues = policyData.final.Q;"),
        ("Math.min(...stateValues)", "Math.min(...stateScores)"),
        ("Math.max(...stateValues)", "Math.max(...stateScores)"),
        ("(stateValues[stateId] - minV) / scale", "(stateScores[stateId] - minV) / scale"),
        ("${stateValues[stateId].toFixed(2)}", "${stateScores[stateId].toFixed(2)}"),
        ("Value V*", "State Score max_a Q(s,a)"),
        ("${stateValues[state].toFixed(3)}", "${stateScores[state].toFixed(3)}"),
        ('<div class="stat-label">Q* Row</div>', '<div class="stat-label">Learned Q Row</div>'),
    ]
    for old, new in replacements:
        template = template.replace(old, new)
    return template


def build_model_template() -> str:
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
        ("const vValues = finalData.V;", "const stateScores = finalData.state_scores;"),
        ('const currentVEl = document.getElementById("current-v");', 'const currentScoreEl = document.getElementById("current-score");'),
        ('const epsilonEl = document.getElementById("epsilon");', 'const numEpisodesEl = document.getElementById("num-episodes");'),
        ('const stableEl = document.getElementById("stable");', 'const alphaEl = document.getElementById("alpha");'),
        ('const actionOrderEl = document.getElementById("action-order");', 'const gammaEl = document.getElementById("gamma");\n    const epsilonStartEl = document.getElementById("epsilon-start");\n    const epsilonFinalEl = document.getElementById("epsilon-final");\n    const actionOrderEl = document.getElementById("action-order");'),
        ("currentVEl.textContent = Number(vValues[selectedState]).toFixed(3);", "currentScoreEl.textContent = Number(stateScores[selectedState]).toFixed(3);"),
        ("epsilonEl.textContent = String(policyData.epsilon);", "numEpisodesEl.textContent = String(policyData.num_episodes);"),
        ("stableEl.textContent = String(finalData.stable);", "alphaEl.textContent = String(policyData.alpha);"),
        ('actionOrderEl.textContent = actionOrder.join(", ");', 'gammaEl.textContent = String(policyData.gamma);\n      epsilonStartEl.textContent = String(policyData.epsilon_start);\n      epsilonFinalEl.textContent = String(policyData.epsilon_final);\n      actionOrderEl.textContent = actionOrder.join(", ");'),
    ]
    for old, new in replacements:
        template = template.replace(old, new)
    return template


def main() -> None:
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
