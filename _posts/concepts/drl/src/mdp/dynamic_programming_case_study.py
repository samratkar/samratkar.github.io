"""Dynamic programming solution for the 3x3 grid-world case study."""

from __future__ import annotations

import numpy as np

from gridworld_case_study import GridWorldCaseStudyEnv, format_policy


def policy_evaluation(env, policy, gamma=0.9, theta=1e-8):
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    while True:
        delta = 0.0
        for s in range(num_states):
            old_v = V[s]
            new_v = 0.0

            for a, action_prob in enumerate(policy[s]):
                if action_prob == 0.0:
                    continue

                for prob, next_state, reward, done in env.P[s][a]:
                    new_v += action_prob * prob * (
                        reward + gamma * (1 - done) * V[next_state]
                    )

            V[s] = new_v
            delta = max(delta, abs(old_v - new_v))

        if delta < theta:
            return V


def compute_q_from_v(env, V, gamma=0.9):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    for s in range(num_states):
        for a in range(num_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                Q[s, a] += prob * (
                    reward + gamma * (1 - done) * V[next_state]
                )

    return Q


def policy_improvement(env, V, gamma=0.9):
    Q = compute_q_from_v(env, V, gamma=gamma)
    num_states, num_actions = Q.shape
    policy = np.zeros((num_states, num_actions))

    for s in range(num_states):
        best_action = int(np.argmax(Q[s]))
        policy[s, best_action] = 1.0

    return policy, Q


def policy_iteration(env, gamma=0.9, theta=1e-8):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.ones((num_states, num_actions)) / num_actions

    while True:
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy, Q = policy_improvement(env, V, gamma=gamma)
        if np.array_equal(new_policy, policy):
            return policy, V, Q
        policy = new_policy


def policy_iteration_with_history(env, gamma=0.9, theta=1e-8):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.ones((num_states, num_actions)) / num_actions
    history = []

    iteration = 0
    while True:
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy, Q = policy_improvement(env, V, gamma=gamma)
        history.append(
            {
                "iteration": iteration,
                "policy": new_policy.copy(),
                "V": V.copy(),
                "Q": Q.copy(),
                "stable": np.array_equal(new_policy, policy),
            }
        )
        if np.array_equal(new_policy, policy):
            return new_policy, V, Q, history
        policy = new_policy
        iteration += 1


def main():
    env = GridWorldCaseStudyEnv()
    policy, V, Q, history = policy_iteration_with_history(env)

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
        print("Greedy policy")
        print(format_policy(snapshot["policy"]))

    print("Optimal state values V*")
    print(V.reshape(3, 3))
    print("\nOptimal action values Q*")
    print(Q)
    print("\nGreedy policy from DP")
    print(format_policy(policy))


if __name__ == "__main__":
    main()
