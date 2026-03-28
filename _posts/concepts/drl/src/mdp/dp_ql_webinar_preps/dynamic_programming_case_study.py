"""Dynamic programming solution for the 3x3 grid-world case study.

End-to-end story of this file:
1. Import the gridworld environment and a helper to print policies.
2. `policy_evaluation()` computes state values `V^pi` for a fixed policy `pi`,
   taking expectation over both:
   - action probabilities `pi(a|s)`
   - transition probabilities `P(s', r | s, a)`
3. `compute_q_from_v()` converts those state values into action values `Q^pi`.
4. `policy_improvement()` converts those action values into an epsilon-greedy
   policy, producing a better policy candidate.
5. `policy_iteration()` alternates evaluation and improvement until the policy
   stops changing, which yields the optimal policy for this finite MDP.
6. `policy_iteration_with_history()` does the same work but also records every
   intermediate snapshot so the learning process can be printed or visualized.
7. `main()` runs the procedure end-to-end and prints the value tables and
   policies for each iteration.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from gridworld_case_study import (
    ACTION_NAMES,
    GridWorldCaseStudyEnv,
    format_greedy_actions,
    format_policy,
)


def format_q_tie_details(Q):
    # Show exact full-precision rows when multiple actions share the same
    # maximal Q-value so tie-breaking is visible instead of being hidden by
    # rounded table output or a single-arrow policy renderer.
    lines = []
    for state, row in enumerate(np.asarray(Q, dtype=float)):
        best_value = row.max()
        tied_actions = np.flatnonzero(np.isclose(row, best_value))
        if len(tied_actions) > 1:
            row_values = ", ".join(f"{value:.15f}" for value in row)
            action_names = ", ".join(
                ACTION_NAMES[int(action)].upper() for action in tied_actions
            )
            lines.append(
                f"state {state}: Q=[{row_values}] tied_best_actions=[{action_names}]"
            )
    return "\n".join(lines) if lines else "No tied greedy actions."


def serialize_transition_model(env):
    transition_model = {}
    for state, actions in env.P.items():
        transition_model[str(state)] = {}
        for action, outcomes in actions.items():
            action_name = ACTION_NAMES[int(action)].upper()
            transition_model[str(state)][action_name] = [
                {
                    "probability": float(prob),
                    "next_state": int(next_state),
                    "reward": float(reward),
                    "done": bool(done),
                }
                for prob, next_state, reward, done in outcomes
            ]
    return transition_model


def export_final_policy_json(
    env,
    policy,
    V,
    Q,
    stable,
    epsilon,
    output_path="dynamic_programming_policies.json",
):
    payload = {
        "epsilon": epsilon,
        "action_order": [
            ACTION_NAMES[i].upper() for i in range(policy.shape[1])
        ],
        "final": {
            "stable": bool(stable),
            "policy_matrix": policy.tolist(),
            "V": V.tolist(),
            "Q": Q.tolist(),
            "transition_model": serialize_transition_model(env),
        },
    }
    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def policy_evaluation(env, policy, gamma=0.9, theta=1e-8, max_iterations=10000):
    # Precondition:
    # - `env` must expose:
    #   - `observation_space.n`
    #   - transition model `P[state][action]`
    # - `policy` must have shape `(num_states, num_actions)`
    # - each row of `policy` should represent action probabilities for one state
    # - `gamma` should be in a sensible discount range, usually 0 <= gamma <= 1
    # - `theta` should be a small positive convergence tolerance
    #
    # What happens:
    # 1. Create a value table `V` initialized to zero for every state.
    # 2. Repeatedly sweep through all states.
    # 3. For each state, compute a new expected value under the current policy:
    #    - first loop over actions weighted by `policy[s][a] = pi(a|s)`
    #    - then, for each action, loop over all transitions in `env.P[s][a]`
    #      weighted by their transition probability
    #    - accumulate the Bellman expectation update over both probability
    #      distributions
    # 4. Track `delta`, the largest absolute value change in this sweep.
    # 5. Stop when the value table changes by less than `theta`.
    #
    # Postcondition:
    # Returns an approximate fixed point `V` for the Bellman expectation
    # equation under the supplied policy.
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    for _ in range(max_iterations):
        delta = 0.0
        for s in range(num_states):
            # Save the previous estimate so convergence can be measured later.
            old_v = V[s]
            new_v = 0.0

            for a, action_prob in enumerate(policy[s]):
                if action_prob == 0.0:
                    # Skip impossible actions under this policy row.
                    continue

                for prob, next_state, reward, done in env.P[s][a]:
                    # Bellman expectation backup:
                    # average over:
                    # - action probability pi(a|s)
                    # - transition probability P(s', r | s, a)
                    # of immediate reward + discounted next-state value
                    new_v += action_prob * prob * (
                        reward + gamma * (1 - done) * V[next_state]
                    )

            V[s] = new_v
            # `delta` stores the largest change seen in this full sweep.
            delta = max(delta, abs(old_v - new_v))

        if delta < theta:
            return V

    raise RuntimeError(
        "policy_evaluation did not converge within max_iterations="
        f"{max_iterations}"
    )


def compute_q_from_v(env, V, gamma=0.9):
    # Why convert V to Q?
    # `V[s]` tells us how good state `s` is overall, but policy improvement
    # must compare individual actions available in that state.
    # `Q[s, a]` answers that finer question:
    # "If I am in state `s`, how good is it to take action `a`?"
    # Once `Q` is available, the best action is identified by `argmax_a Q[s, a]`.
    # Precondition:
    # - `env` must expose `observation_space.n`, `action_space.n`, and `P`
    # - `V` must contain one value per state
    # - `gamma` should be a valid discount factor
    #
    # What happens:
    # 1. Allocate an empty Q-table with one row per state and one column per action.
    # 2. For each state-action pair, sum over all transitions in `env.P[s][a]`.
    #    This means `Q[s, a]` averages over the transition probabilities in
    #    `P`, but not over action probabilities from `pi`, because the action
    #    `a` is assumed to already be fixed.
    # 3. Apply the Bellman one-step lookahead using the provided state values `V`.
    #
    # Postcondition:
    # Returns a Q-table where `Q[s, a]` estimates the value of taking action `a`
    # in state `s` and then following the value function encoded in `V`.
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    for s in range(num_states):
        for a in range(num_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                # One-step lookahead from V to Q.
                Q[s, a] += prob * (
                    reward + gamma * (1 - done) * V[next_state]
                )

    return Q


def policy_improvement(env, V, gamma=0.9, epsilon=0.1):
    # Why this step needs Q instead of only V:
    # `V` gives one value per state, but improvement must choose among multiple
    # actions in each state. So we first compute `Q[s, a]` for all actions and
    # then assign most of the probability mass to the best action.
    # Precondition:
    # - `env` must be a valid tabular environment
    # - `V` must contain one value per state
    # - `gamma` should be a valid discount factor
    # - `epsilon` should satisfy 0 <= epsilon <= 1
    #
    # What happens:
    # 1. Convert the current state values `V` into action values `Q`.
    # 2. For each state, find all actions tied for the largest Q-value.
    # 3. Build an epsilon-greedy policy:
    #    - distribute `epsilon` uniformly across all actions
    #    - split the remaining `1 - epsilon` probability equally across all
    #      tied best actions
    #    This produces a policy row representing action probabilities
    #    `pi(a|s)` without arbitrarily breaking ties among equally good actions.
    #
    # Postcondition:
    # Returns:
    # - `policy`: an epsilon-greedy policy with shape `(num_states, num_actions)`
    # - `Q`: the action-value table used to choose that policy
    Q = compute_q_from_v(env, V, gamma=gamma)
    num_states, num_actions = Q.shape
    policy = np.full((num_states, num_actions), epsilon / num_actions)

    for s in range(num_states):
        row = np.asarray(Q[s], dtype=float)
        best_value = row.max()
        best_actions = np.flatnonzero(np.isclose(row, best_value))
        greedy_share = (1.0 - epsilon) / len(best_actions)
        policy[s, best_actions] += greedy_share

    return policy, Q


def policy_iteration(
    env, gamma=0.9, theta=1e-8, epsilon=0.1, max_policy_iterations=1000
):
    # Precondition:
    # - `env` must be a finite tabular MDP
    # - `gamma`, `theta`, and `epsilon` should be valid numeric hyperparameters
    #
    # What happens:
    # 1. Start from a uniform random policy.
    # 2. Evaluate that policy to get `V`.
    # 3. Improve the policy using an epsilon-greedy distribution derived from `V`.
    # 4. If the policy no longer changes, stop.
    # 5. Otherwise repeat evaluation and improvement.
    #
    # Postcondition:
    # Returns a stable policy along with its state-value and action-value tables.
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.ones((num_states, num_actions)) / num_actions

    for _ in range(max_policy_iterations):
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy, Q = policy_improvement(env, V, gamma=gamma, epsilon=epsilon)
        stable = np.array_equal(
            new_policy.argmax(axis=1), policy.argmax(axis=1)
        )
        # Stability means the greedy action chosen in each state no longer changes.
        if stable:
            return policy, V, Q
        policy = new_policy

    raise RuntimeError(
        "policy_iteration did not converge within max_policy_iterations="
        f"{max_policy_iterations}"
    )


def policy_iteration_with_history(
    env, gamma=0.9, theta=1e-8, epsilon=0.1, max_policy_iterations=1000
):
    # Precondition:
    # Same as `policy_iteration()`.
    #
    # What happens:
    # 1. Run the same evaluation/improvement loop as policy iteration.
    # 2. After each improvement step, record a snapshot containing:
    #    - iteration index
    #    - policy
    #    - state values `V`
    #    - action values `Q`
    #    - whether the policy is already stable
    # 3. Stop once stability is reached.
    #
    # Postcondition:
    # Returns the final policy, `V`, `Q`, and a history list of intermediate
    # snapshots for debugging, teaching, or visualization.
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.ones((num_states, num_actions)) / num_actions
    history = []

    iteration = 0
    for _ in range(max_policy_iterations):
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy, Q = policy_improvement(env, V, gamma=gamma, epsilon=epsilon)
        stable = np.array_equal(
            new_policy.argmax(axis=1), policy.argmax(axis=1)
        )
        history.append(
            {
                # Iteration number in the outer policy-iteration loop.
                "iteration": iteration,
                # Copy arrays so later updates do not overwrite past snapshots.
                "policy": new_policy.copy(),
                "V": V.copy(),
                "Q": Q.copy(),
                "stable": stable,
            }
        )
        if stable:
            return new_policy, V, Q, history
        policy = new_policy
        iteration += 1

    raise RuntimeError(
        "policy_iteration_with_history did not converge within "
        f"max_policy_iterations={max_policy_iterations}"
    )


def main():
    # Precondition:
    # This script is being run as a program and NumPy plus the local gridworld
    # module are available.
    #
    # What happens:
    # 1. Construct the gridworld environment.
    # 2. Run policy iteration while collecting history.
    # 3. Configure NumPy printing for compact readable tables.
    # 4. Print each intermediate snapshot.
    # 5. Print the final state values, action values, and epsilon-greedy policy.
    #
    # Postcondition:
    # The dynamic-programming solution and its intermediate steps are printed
    # to standard output.
    env = GridWorldCaseStudyEnv()
    epsilon = 0.1
    policy, V, Q, history = policy_iteration_with_history(env, epsilon=epsilon)

    np.set_printoptions(precision=3, suppress=True)
    print("Dynamic Programming Snapshots")
    for snapshot in history:
        print(
            f"\nIteration {snapshot['iteration']} "
            f"(stable={snapshot['stable']})"
        )
        print("V")
        print(snapshot["V"].reshape(3, 3))
        print("Q")
        print(snapshot["Q"])
        print("Full-precision tied greedy Q rows")
        print(format_q_tie_details(snapshot["Q"]))
        print("Greedy actions from Q (ties shown)")
        print(format_greedy_actions(snapshot["Q"]))
        print(f"Epsilon-greedy policy (epsilon={epsilon})")
        print(format_policy(snapshot["policy"]))

    print("Optimal state values V*")
    print(V.reshape(3, 3))
    print("\nOptimal action values Q*")
    print(Q)
    print("\nFull-precision tied greedy Q rows")
    print(format_q_tie_details(Q))
    print("\nGreedy actions from Q* (ties shown)")
    print(format_greedy_actions(Q))
    print(f"\nEpsilon-greedy policy from DP (epsilon={epsilon})")
    print(format_policy(policy))
    export_path = export_final_policy_json(
        env=env,
        policy=policy,
        V=V,
        Q=Q,
        stable=history[-1]["stable"] if history else True,
        epsilon=epsilon,
    )
    print(f"\nExported final policy JSON to {export_path.resolve()}")


if __name__ == "__main__":
    main()
