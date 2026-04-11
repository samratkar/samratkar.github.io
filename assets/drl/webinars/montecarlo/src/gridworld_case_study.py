"""Shared 3x3 grid-world used by the Monte Carlo case study."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTION_NAMES = {
    UP: "UP",
    RIGHT: "RIGHT",
    DOWN: "DOWN",
    LEFT: "LEFT",
}


@dataclass(frozen=True)
class GridWorldConfig:
    rows: int = 3
    cols: int = 3
    start_state: int = 0
    goal_state: int = 8
    step_reward: float = -1.0
    goal_reward: float = 10.0
    intended_move_prob: float = 0.8
    slip_left_prob: float = 0.1
    slip_right_prob: float = 0.1


class GridWorldCaseStudyEnv:
    def __init__(self, config: GridWorldConfig | None = None, seed: int = 7) -> None:
        self.config = config or GridWorldConfig()
        self.observation_space = SimpleNamespace(n=self.config.rows * self.config.cols)
        self.action_space = SimpleNamespace(n=4)
        self.P = self._build_transition_model()
        self.rng = np.random.default_rng(seed)
        self.state = self.config.start_state

    def _build_transition_model(self) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        transitions: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
        for state in range(self.observation_space.n):
            transitions[state] = {}
            for action in range(self.action_space.n):
                transitions[state][action] = self._transition_distribution(state, action)
        return transitions

    def _transition_distribution(self, state: int, action: int) -> List[Tuple[float, int, float, bool]]:
        if state == self.config.goal_state:
            return [(1.0, state, 0.0, True)]

        candidate_actions = [
            (action, self.config.intended_move_prob),
            ((action - 1) % self.action_space.n, self.config.slip_left_prob),
            ((action + 1) % self.action_space.n, self.config.slip_right_prob),
        ]
        aggregated_probs: Dict[int, float] = {}
        for candidate_action, prob in candidate_actions:
            if prob == 0.0:
                continue
            next_state = self._move(state, candidate_action)
            aggregated_probs[next_state] = aggregated_probs.get(next_state, 0.0) + prob

        outcomes: List[Tuple[float, int, float, bool]] = []
        for next_state, prob in aggregated_probs.items():
            done = next_state == self.config.goal_state
            reward = self.config.goal_reward if done else self.config.step_reward
            outcomes.append((prob, next_state, reward, done))
        return outcomes

    def _move(self, state: int, action: int) -> int:
        if state == self.config.goal_state:
            return state

        row, col = divmod(state, self.config.cols)
        next_row, next_col = row, col
        if action == UP:
            next_row = max(0, row - 1)
        elif action == RIGHT:
            next_col = min(self.config.cols - 1, col + 1)
        elif action == DOWN:
            next_row = min(self.config.rows - 1, row + 1)
        elif action == LEFT:
            next_col = max(0, col - 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        return next_row * self.config.cols + next_col

    def reset(self) -> Tuple[int, dict]:
        self.state = self.config.start_state
        return self.state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        outcomes = self.P[self.state][action]
        probabilities = [prob for prob, _, _, _ in outcomes]
        outcome_index = int(self.rng.choice(len(outcomes), p=probabilities))
        _, next_state, reward, done = outcomes[outcome_index]
        self.state = next_state
        return next_state, reward, done, False, {}
