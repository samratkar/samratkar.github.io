"""Shared 3x3 grid-world used by the DP vs Q-learning case study.

End-to-end story of this file:
1. `GridWorldConfig` defines the static rules of the world:
   grid size, start state, goal state, rewards, and transition probabilities.
2. `GridWorldCaseStudyEnv.__init__()` uses that config to build a small
   Gym-like environment object with:
   - `observation_space.n`: how many discrete states exist
   - `action_space.n`: how many discrete actions exist
   - `P`: the tabular transition model for the MDP
   - `state`: the current live state while an episode is running
3. `_build_transition_model()` precomputes every `(state, action)` transition
   by calling `_transition_distribution()` for all state-action pairs.
4. `_transition_distribution()` contains the actual environment dynamics:
   given one state and one action, it builds the full probability distribution
   over next states and rewards.
5. `reset()` starts a new episode by placing the agent back at the start state.
6. `step(action)` advances the environment by one action using the precomputed
   transition model `P` by sampling from that transition distribution.
7. `format_policy(policy)` is a helper for visualization; it converts a policy
   table into a text grid with arrows.
"""

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
    UP: "Up",
    RIGHT: "Right",
    DOWN: "Down",
    LEFT: "Left",
}


@dataclass(frozen=True)
class GridWorldConfig:
    """Immutable parameter bundle for the case-study grid world.

    Story:
    This dataclass describes the static MDP specification rather than the live
    runtime environment. It holds the grid geometry, the start and goal states,
    the reward scheme, and the stochastic movement probabilities used when the
    agent chooses an action.

    Precondition:
    The field values should describe a valid grid world. In particular, the
    start and goal states should refer to cells inside the `rows x cols` grid,
    and the movement probabilities should form a sensible distribution.

    Postcondition:
    A frozen configuration object exists and can be safely shared across
    environments, notebooks, and rendering scripts without accidental mutation.
    """
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
    """Minimal tabular environment with a Gym-like interface.

    Story:
    This class packages the full MDP environment:
    - static settings in `config`
    - the set of possible states in `observation_space`
    - the set of possible actions in `action_space`
    - the transition model in `P`
    - the current episode state in `state`
    """

    # Declare the main instance attributes in one place so the class layout is
    config: GridWorldConfig
    observation_space: SimpleNamespace
    action_space: SimpleNamespace
    P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]
    state: int
    rng: np.random.Generator

    def __init__(self, config: GridWorldConfig | None = None) -> None:
        """Initialize the environment and precompute its tabular dynamics.

        Precondition:
        `config` is either a caller-supplied `GridWorldConfig` or `None`, in
        which case the default case-study configuration is used.

        What happens:
        1. Resolve the environment configuration.
        2. Create lightweight Gym-like `observation_space` and `action_space`
           objects exposing the number of states and actions.
        3. Build the full transition table `P` for every `(state, action)` pair.
        4. Create a random number generator used by `step()`.
        5. Set the live state to the configured start state.

        Postcondition:
        The environment is ready for repeated `reset()` and `step()` calls.
        `P[state][action]` stores a list of transition tuples in the form
        `(probability, next_state, reward, done)`.
        """
        self.config = config or GridWorldConfig()
        # Gym exposes an observation space object. For a discrete tabular MDP,
        # the only field needed here is `n`: the total number of valid states.
        self.observation_space = SimpleNamespace(
            n=self.config.rows * self.config.cols
        )
        # Gym also exposes an action space object. This environment uses four
        # discrete actions, encoded by the integer constants defined above.
        self.action_space = SimpleNamespace(n=4)
        # `P` is the classic tabular transition model used by many Gym toy-text
        # environments.
        # The `(state, action)` pair is represented by the nested dictionary
        # lookup `P[state][action]`.
        # The value at that lookup is a list of all possible outcomes for the
        # state-transition distribution P(s', r, done | s, a).
        # Each outcome tuple is:
        # (probability, next_state, reward, done)
        # The number of items in this inner list is the number of distinct
        # probability-mass entries in the transition distribution for the chosen
        # state-action pair.
        # In this version the list can contain multiple outcomes, so one
        # `(state, action)` pair can represent a full stochastic transition
        # distribution rather than a single deterministic move.
        self.P = self._build_transition_model()
        self.rng = np.random.default_rng()
        # Mutable runtime state storing the agent's current position.
        # `reset()` reinitializes it and `step()` advances it.
        self.state = self.config.start_state

    def _build_transition_model(
        self,
    ) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        """Build the full tabular transition model for the environment.

        Precondition:
        `self.config`, `self.observation_space`, and `self.action_space` are
        already initialized.

        What happens:
        1. Allocate an empty nested dictionary.
        2. Iterate over every discrete state.
        3. For each state, iterate over every discrete action.
        4. Compute the complete stochastic outcome distribution for that pair.
        5. Store the resulting list at `transitions[state][action]`.

        Postcondition:
        Returns a complete tabular model where the outer key is the current
        state, the inner key is the action, and the value is the list of
        environment outcomes for that state-action pair.
        """
        transitions: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
        for state in range(self.observation_space.n):
            transitions[state] = {}
            for action in range(self.action_space.n):
                transitions[state][action] = self._transition_distribution(
                    state, action
                )
        return transitions

    def _transition_distribution(
        self, state: int, action: int
    ) -> List[Tuple[float, int, float, bool]]:
        """Compute the stochastic outcomes produced by one state-action pair.

        Precondition:
        `state` is a valid state index and `action` is one of the directional
        action constants `{UP, RIGHT, DOWN, LEFT}`.

        What happens:
        1. If the agent is already at the goal, return a self-loop terminal
           transition with probability 1.
        2. Otherwise form the intended move plus the slip-left and slip-right
           alternatives using the configured movement probabilities.
        3. Apply `_move()` to each candidate direction.
        4. Aggregate probabilities when different move branches land in the
           same next state.
        5. Attach the appropriate reward and terminal flag to each outcome.

        Postcondition:
        Returns the tabular transition distribution as a list of
        `(probability, next_state, reward, done)` tuples.
        """
        if state == self.config.goal_state:
            # Terminal states loop to themselves with probability 1.
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
        """Apply one directional move while clamping the result to the grid.

        Precondition:
        `state` is a valid discrete state index and `action` is one of the four
        directional action constants.

        What happens:
        1. Convert the flat state id into `(row, col)` coordinates.
        2. Apply the requested directional move.
        3. Clamp the move so the agent cannot leave the grid.
        4. Convert the resulting coordinates back into a flat state id.

        Postcondition:
        Returns the single next state induced by that movement direction.
        """
        if state == self.config.goal_state:
            return state

        # States are stored as a single integer, but movement logic is easier in
        # (row, col) form. `divmod` converts the flat state id into grid coords.
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

        # Convert the next grid cell back into the flattened discrete state id.
        return next_row * self.config.cols + next_col

    def reset(self) -> Tuple[int, dict]:
        """Reset the live episode state to the configured start cell.

        Precondition:
        The environment has already been constructed.

        What happens:
        1. Discard any progress from the current episode.
        2. Restore `self.state` to `self.config.start_state`.

        Postcondition:
        Returns the Gym-style pair `(start_state, {})` and leaves the
        environment ready for a fresh episode rollout.
        """
        self.state = self.config.start_state
        return self.state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Advance the environment by sampling one outcome from `P[self.state][action]`.

        Precondition:
        `action` is a valid discrete action id, `self.state` contains a valid
        current state, and the transition model `P` has already been built.

        What happens:
        1. Look up all transition outcomes for the current state and action.
        2. Sample one outcome according to its stored probabilities.
        3. Extract the sampled next state, reward, and terminal flag.
        4. Update the live state to the sampled next state.
        5. Return the Gym-like result tuple.

        Postcondition:
        Returns `(next_state, reward, terminated, truncated, info)` where
        `truncated` is always `False` and `info` is an empty dictionary for
        this simplified environment.
        """
        outcomes = self.P[self.state][action]
        probabilities = [prob for prob, _, _, _ in outcomes]
        outcome_index = int(self.rng.choice(len(outcomes), p=probabilities))
        _, next_state, reward, done = outcomes[outcome_index]
        self.state = next_state
        return next_state, reward, done, False, {}


def _format_action_grid(rows) -> str:
    """Render a flat list of cell labels as a boxed 3x3 text grid."""
    lines = []
    cols = GridWorldConfig().cols
    separator = "+---+---+---+"
    lines.append(separator)
    for idx in range(0, len(rows), cols):
        lines.append("|" + "|".join(rows[idx : idx + cols]) + "|")
        lines.append(separator)
    return "\n".join(lines)


def format_policy(policy) -> str:
    """Format a policy matrix as a text grid of greedy-action arrows.

    Precondition:
    `policy` is indexable by state and each row contains action probabilities
    for that state.

    What happens:
    1. Find the highest-probability action or actions in each state.
    2. Map those greedy actions to arrow symbols.
    3. Preserve ties by rendering every tied action in the same cell.
    4. Mark the goal cell with `G`.

    Postcondition:
    Returns a multi-line string suitable for notebook output or debugging.
    """
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
            row = np.asarray(policy[state], dtype=float)
            best_value = row.max()
            best_actions = np.flatnonzero(np.isclose(row, best_value))
            cell = "".join(arrows[action] for action in best_actions)
            rows.append(f"{cell:^3}")

    return _format_action_grid(rows)


def format_greedy_actions(Q) -> str:
    """Format the greedy action set implied by a Q-table as a text grid.

    Precondition:
    `Q` is a state-action value table with one row per state.

    What happens:
    1. Compute the maximal Q-value in each state.
    2. Identify every action tied for that maximum.
    3. Render all greedy actions in the cell so ties remain visible.
    4. Mark the goal cell with `G`.

    Postcondition:
    Returns a multi-line string that exposes greedy-action ties directly from
    the learned or computed Q-values.
    """
    arrows = {
        UP: "^",
        RIGHT: ">",
        DOWN: "v",
        LEFT: "<",
    }
    rows = []
    for state in range(Q.shape[0]):
        if state == GridWorldConfig().goal_state:
            rows.append(" G ")
        else:
            row = np.asarray(Q[state], dtype=float)
            best_value = row.max()
            best_actions = np.flatnonzero(np.isclose(row, best_value))
            cell = "".join(arrows[action] for action in best_actions)
            rows.append(f"{cell:^3}")

    return _format_action_grid(rows)
