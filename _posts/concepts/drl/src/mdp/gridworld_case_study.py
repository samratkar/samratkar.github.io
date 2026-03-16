"""Shared 3x3 grid-world used by the DP vs Q-learning case study."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTION_NAMES = {
    UP: "Up",
    RIGHT: "Right",
    DOWN: "Down",
    LEFT: "Left",
}


@dataclass(frozen=True)
class GridWorldConfig:
    rows: int = 3
    cols: int = 3
    start_state: int = 0
    goal_state: int = 8
    step_reward: float = -1.0
    goal_reward: float = 10.0


class GridWorldCaseStudyEnv:
    """Minimal tabular environment with a Gym-like interface."""

    def __init__(self, config: GridWorldConfig | None = None) -> None:
        self.config = config or GridWorldConfig()
        self.observation_space = SimpleNamespace(
            n=self.config.rows * self.config.cols
        )
        self.action_space = SimpleNamespace(n=4)
        self.P = self._build_transition_model()
        self.state = self.config.start_state

    def _build_transition_model(
        self,
    ) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        transitions: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
        for state in range(self.observation_space.n):
            transitions[state] = {}
            for action in range(self.action_space.n):
                transitions[state][action] = [self._transition(state, action)]
        return transitions

    def _transition(self, state: int, action: int) -> Tuple[float, int, float, bool]:
        if state == self.config.goal_state:
            return (1.0, state, 0.0, True)

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

        next_state = next_row * self.config.cols + next_col
        done = next_state == self.config.goal_state
        reward = self.config.goal_reward if done else self.config.step_reward
        return (1.0, next_state, reward, done)

    def reset(self) -> Tuple[int, dict]:
        self.state = self.config.start_state
        return self.state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        _, next_state, reward, done = self.P[self.state][action][0]
        self.state = next_state
        return next_state, reward, done, False, {}


def format_policy(policy) -> str:
    arrows = {
        UP: "^",
        RIGHT: ">",
        DOWN: "v",
        LEFT: "<",
    }
    rows = []
    for state in range(policy.shape[0]):
        if state == GridWorldConfig().goal_state:
            rows.append(" G ")
        else:
            rows.append(f" {arrows[int(policy[state].argmax())]} ")

    lines = []
    cols = GridWorldConfig().cols
    separator = "+---+---+---+"
    lines.append(separator)
    for idx in range(0, len(rows), cols):
        lines.append("|" + "|".join(rows[idx : idx + cols]) + "|")
        lines.append(separator)
    return "\n".join(lines)
