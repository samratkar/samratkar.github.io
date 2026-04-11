"""First-visit on-policy Monte Carlo control for the shared 3x3 grid world."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gridworld_case_study import ACTION_NAMES, GridWorldCaseStudyEnv


def epsilon_greedy_distribution(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    num_states, num_actions = q_values.shape
    policy = np.full((num_states, num_actions), epsilon / num_actions, dtype=float)
    best_actions = np.argmax(q_values, axis=1)
    policy[np.arange(num_states), best_actions] += 1.0 - epsilon
    return policy


def choose_action(policy_row: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(policy_row), p=policy_row))


def generate_episode(env: GridWorldCaseStudyEnv, policy: np.ndarray, max_steps: int) -> List[Tuple[int, int, float]]:
    state, _ = env.reset()
    trajectory: List[Tuple[int, int, float]] = []
    for _ in range(max_steps):
        action = choose_action(policy[state], env.rng)
        next_state, reward, terminated, _, _ = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        if terminated:
            break
    return trajectory


def run_first_visit_mc_control(
    num_episodes: int = 7000,
    gamma: float = 0.9,
    epsilon_start: float = 0.25,
    epsilon_final: float = 0.02,
    max_steps_per_episode: int = 50,
    seed: int = 7,
) -> Dict[str, object]:
    env = GridWorldCaseStudyEnv(seed=seed)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_values = np.zeros((num_states, num_actions), dtype=float)
    returns_sum = np.zeros((num_states, num_actions), dtype=float)
    returns_count = np.zeros((num_states, num_actions), dtype=float)
    rng = np.random.default_rng(seed)
    env.rng = rng

    episode_returns: List[float] = []
    checkpoints: List[Dict[str, object]] = []
    checkpoint_targets = {1, 10, 100, 1000, num_episodes}

    for episode_idx in range(1, num_episodes + 1):
        progress = (episode_idx - 1) / max(1, num_episodes - 1)
        epsilon = epsilon_start + (epsilon_final - epsilon_start) * progress
        policy = epsilon_greedy_distribution(q_values, epsilon)
        policy[env.config.goal_state] = np.full(num_actions, 1.0 / num_actions)
        trajectory = generate_episode(env, policy, max_steps_per_episode)

        discounted_return = 0.0
        visited_pairs = set()
        total_episode_reward = sum(reward for _, _, reward in trajectory)

        for step_idx in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[step_idx]
            discounted_return = gamma * discounted_return + reward
            if (state, action) in visited_pairs:
                continue
            visited_pairs.add((state, action))
            returns_sum[state, action] += discounted_return
            returns_count[state, action] += 1.0
            q_values[state, action] = returns_sum[state, action] / returns_count[state, action]

        episode_returns.append(total_episode_reward)
        if episode_idx in checkpoint_targets:
            checkpoints.append({
                "episode": episode_idx,
                "epsilon": float(epsilon),
                "state_scores": np.max(q_values, axis=1).round(6).tolist(),
                "greedy_actions": np.argmax(q_values, axis=1).astype(int).tolist(),
            })

    final_policy = epsilon_greedy_distribution(q_values, epsilon_final)
    final_policy[env.config.goal_state] = np.full(num_actions, 1.0 / num_actions)

    transition_model = {
        str(state): {
            ACTION_NAMES[action]: [
                {
                    "probability": float(prob),
                    "next_state": int(next_state),
                    "reward": float(reward),
                    "done": bool(done),
                }
                for prob, next_state, reward, done in env.P[state][action]
            ]
            for action in range(num_actions)
        }
        for state in range(num_states)
    }

    preview_episodes = []
    preview_policy = epsilon_greedy_distribution(q_values, 0.05)
    preview_policy[env.config.goal_state] = np.full(num_actions, 1.0 / num_actions)
    for _ in range(3):
        episode = generate_episode(env, preview_policy, max_steps_per_episode)
        preview_episodes.append([
            {
                "state": int(state),
                "action": ACTION_NAMES[action],
                "reward": float(reward),
            }
            for state, action, reward in episode
        ])

    moving_average_window = 100
    moving_average = []
    for idx in range(len(episode_returns)):
        start = max(0, idx - moving_average_window + 1)
        moving_average.append(float(np.mean(episode_returns[start : idx + 1])))

    return {
        "algorithm": "first_visit_on_policy_monte_carlo_control",
        "seed": seed,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_final": epsilon_final,
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "action_order": [ACTION_NAMES[action] for action in range(num_actions)],
        "environment": asdict(env.config),
        "checkpoints": checkpoints,
        "preview_episodes": preview_episodes,
        "training_curve": {
            "episode_returns": [float(value) for value in episode_returns],
            "moving_average": moving_average,
        },
        "final": {
            "policy_matrix": final_policy.tolist(),
            "Q": q_values.tolist(),
            "state_scores": np.max(q_values, axis=1).tolist(),
            "visit_counts": returns_count.tolist(),
            "transition_model": transition_model,
        },
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "monte_carlo_policies.json"
    payload = run_first_visit_mc_control()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
