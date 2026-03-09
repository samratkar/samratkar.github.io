"""
Markov Decision Process with Gymnasium environment.
"""

import csv
import gymnasium as gym


env = gym.make('FrozenLake-v1', is_slippery=True)
print("observation space = ", env.observation_space)
print("action space = ", env.action_space)
num_actions = env.action_space.n
num_states = env.observation_space.n
print("Number of actions:", num_actions)
print("Number of states:", num_states)

# Every state has a list of possible actions. Each action has one or more possible
# transitions represented as: (transition_prob, next_state, reward, is_terminal).
rows = []
action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
branch_names = {1: "left_of_intended", 2: "intended", 3: "right_of_intended"}
for state in range(num_states):
    for action in range(num_actions):
        transitions = env.unwrapped.P[state][action]
        for t_idx, (transition_prob, next_state, reward, is_terminal) in enumerate(
            transitions,
            start=1,
        ):
            rows.append(
                {
                    "state": state,
                    "action": action,
                    "action_name": action_names.get(action, str(action)),
                    "transition_no": t_idx,
                    "transition_name": branch_names.get(t_idx, f"branch_{t_idx}"),
                    "transition_prob": transition_prob,
                    "next_state": next_state,
                    "reward": reward,
                    "is_terminal": is_terminal,
                }
            )

headers = [
    "state",
    "action",
    "action_name",
    "transition_no",
    "transition_name",
    "transition_prob",
    "next_state",
    "reward",
    "is_terminal",
]

# Print as an aligned table with a single header row.
widths = {h: len(h) for h in headers}
for row in rows:
    for h in headers:
        widths[h] = max(widths[h], len(str(row[h])))

separator = "+-" + "-+-".join("-" * widths[h] for h in headers) + "-+"
header_line = "| " + " | ".join(f"{h:<{widths[h]}}" for h in headers) + " |"

print("\nMDP Transition Table (env.unwrapped.P)")
print(separator)
print(header_line)
print(separator)
for row in rows:
    print("| " + " | ".join(f"{str(row[h]):<{widths[h]}}" for h in headers) + " |")
print(separator)

# Save the same table rows to CSV.
csv_path = "mdp_unwrapped_transitions.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV exported: {csv_path}")
