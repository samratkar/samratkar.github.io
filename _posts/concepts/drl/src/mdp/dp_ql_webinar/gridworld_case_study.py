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
    # Story:
    # This is a pure data container. It does not execute the environment;
    # it only stores the fixed parameters needed to construct one.
    #
    # Precondition:
    # The values should describe a valid gridworld. In particular, start and
    # goal state indices should be within the total number of cells.
    #
    # Postcondition:
    # A read-only configuration object exists and can be safely shared.
    #
    # Immutable configuration bundle for the environment.
    # This is the structural metadata a Gym-style env usually keeps:
    # grid size, start/goal states, reward settings, and the probability of
    # each possible movement outcome after an action is chosen.
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
        # Precondition:
        # `config` is either:
        # - a `GridWorldConfig` object supplied by the caller, or
        # - `None`, meaning the default config should be used.
        #
        # What happens:
        # 1. Choose the provided config, or create a default one.
        # 2. Build Gym-like `observation_space` and `action_space` objects.
        # 3. Precompute the full transition table `P`.
        # 4. Create a random-number generator for stochastic sampling in `step()`.
        # 5. Initialize the live state to the start state.
        #
        # Postcondition:
        # The environment is ready to be used with `reset()` and `step()`.
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
        # Precondition:
        # `self.config`, `self.observation_space`, and `self.action_space` have
        # already been initialized.
        #
        # What happens:
        # 1. Allocate an empty nested dictionary `transitions`.
        # 2. Loop over every state `s`.
        # 3. For each state, loop over every action `a`.
        # 4. Call `_transition_distribution(s, a)` to compute all stochastic
        #    outcomes for that state-action pair.
        # 5. Store the full probability distribution at `transitions[s][a]`.
        #
        # Postcondition:
        # Returns a complete tabular model of the MDP where every valid
        # state-action pair has an entry.
        # Nested dictionary structure:
        # - outer key: current state id `s`
        # - inner key: action id `a`
        # - `transitions[s][a]` is the list of outcomes in the transition
        #   probability distribution for that state-action pair
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
        # Precondition:
        # - `state` must be a valid discrete state index.
        # - `action` must be one of {UP, RIGHT, DOWN, LEFT}.
        #
        # What happens:
        # 1. If `state` is already the goal state, keep the agent there and mark
        #    the transition as terminal.
        # 2. Otherwise compute three candidate directions:
        #    intended move, slip-left move, and slip-right move.
        # 3. Use the probabilities from `GridWorldConfig` to assign probability
        #    mass to those candidate movement directions.
        # 4. Convert each candidate move into a next state.
        # 5. If multiple outcomes land in the same next state, merge their
        #    probabilities into one probability-mass entry.
        # 6. Attach reward and terminal status to each merged outcome.
        #
        # Postcondition:
        # Returns the full transition distribution for `(state, action)`.
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
        # Precondition:
        # - `state` must be a valid discrete state index.
        # - `action` must be one of the four directional actions.
        #
        # What happens:
        # 1. Convert the flat state index into `(row, col)`.
        # 2. Apply the chosen directional move.
        # 3. Clamp movement to stay inside the grid boundaries.
        # 4. Convert the resulting grid cell back into a flat state index.
        #
        # Postcondition:
        # Returns the next discrete state reached by that single movement.
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
        # Precondition:
        # The environment object has been constructed.
        #
        # What happens:
        # 1. Discard any previous episode progress.
        # 2. Set the live state back to the configured start state.
        #
        # Postcondition:
        # - `self.state == self.config.start_state`
        # - returns `(start_state, {})`
        # - the environment is ready for a fresh episode
        # Gym-style reset returns the initial observation plus an info dict.
        self.state = self.config.start_state
        return self.state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        # Precondition:
        # - `action` must be a valid discrete action id.
        # - `self.state` must currently hold a valid state index.
        # - The transition table `P` must already exist.
        #
        # What happens:
        # 1. Use the current state and chosen action to look up `P[self.state][action]`.
        # 2. Read the full list of possible transition outcomes.
        # 3. Sample one outcome according to its probability.
        # 4. Extract `next_state`, `reward`, and `done`.
        # 5. Update the live environment state to `next_state`.
        # 6. Return the Gym-like step result tuple.
        #
        # Postcondition:
        # - `self.state` is updated to the returned `next_state`
        # - caller receives `(next_state, reward, terminated, truncated, info)`
        # - `truncated` is always `False` in this simple environment
        # Execute one action in the environment.
        # In RL terms this maps:
        # current state s + chosen action a -> next state s', reward, terminal flag
        # `P[self.state][action]` can contain multiple transition tuples, one
        # for each support point in the stochastic transition distribution.
        # Sample one of those outcomes from the tabular model `P`,
        # update the live state, and return the standard Gym-like step tuple:
        # (observation, reward, terminated, truncated, info)
        # `info` is an optional dictionary for extra metadata; it is empty here.
        outcomes = self.P[self.state][action]
        probabilities = [prob for prob, _, _, _ in outcomes]
        outcome_index = int(self.rng.choice(len(outcomes), p=probabilities))
        _, next_state, reward, done = outcomes[outcome_index]
        self.state = next_state
        return next_state, reward, done, False, {}


def format_policy(policy) -> str:
    # Precondition:
    # `policy` should be indexable by state and each row should support
    # `.argmax()` to pick the best action for that state.
    #
    # What happens:
    # 1. Map the best action in each state to an arrow symbol.
    # 2. Replace the goal state's arrow with `G`.
    # 3. Group the states row by row into a text grid.
    #
    # Postcondition:
    # Returns a multi-line string that visually represents the policy.
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
