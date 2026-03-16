"""Q-learning solution for the 3x3 grid-world case study."""

from __future__ import annotations

import random

import numpy as np

from gridworld_case_study import GridWorldCaseStudyEnv, format_policy


def epsilon_greedy_action(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(Q.shape[1])
    return int(np.argmax(Q[state]))


def q_learning(
    env,
    num_episodes=5000,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    max_steps_per_episode=50,
    seed=7,
):
    random.seed(seed)
    np.random.seed(seed)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        for _ in range(max_steps_per_episode):
            action = epsilon_greedy_action(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward + gamma * (1 - done) * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q


def q_learning_with_history(
    env,
    num_episodes=5000,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    max_steps_per_episode=50,
    seed=7,
    snapshot_episodes=None,
):
    random.seed(seed)
    np.random.seed(seed)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    snapshots = []

    if snapshot_episodes is None:
        snapshot_episodes = [1, 10, 100, 500, 1000, num_episodes]
    snapshot_set = set(snapshot_episodes)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False

        for _ in range(max_steps_per_episode):
            action = epsilon_greedy_action(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward + gamma * (1 - done) * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state

            if done:
                break

        if episode in snapshot_set:
            snapshots.append(
                {
                    "episode": episode,
                    "epsilon": epsilon,
                    "Q": Q.copy(),
                    "policy": greedy_policy_from_q(Q),
                }
            )

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, snapshots


def greedy_policy_from_q(Q):
    num_states, num_actions = Q.shape
    policy = np.zeros((num_states, num_actions))

    for s in range(num_states):
        best_action = int(np.argmax(Q[s]))
        policy[s, best_action] = 1.0

    return policy


def main():
    env = GridWorldCaseStudyEnv()
    Q, snapshots = q_learning_with_history(env)
    policy = greedy_policy_from_q(Q)

    np.set_printoptions(precision=3, suppress=True)
    print("Q-Learning Snapshots")
    for snapshot in snapshots:
        print(
            f"\nEpisode {snapshot['episode']} "
            f"(epsilon={snapshot['epsilon']:.3f})"
        )
        print("Q")
        print(snapshot["Q"])
        print("max_a Q(s, a)")
        print(snapshot["Q"].max(axis=1).reshape(3, 3))
        print("Greedy policy")
        print(format_policy(snapshot["policy"]))

    print("Learned action values Q")
    print(Q)
    print("\nGreedy policy from Q-learning")
    print(format_policy(policy))


if __name__ == "__main__":
    main()
