"""Compare dynamic programming and Q-learning on the 3x3 case-study grid."""

from __future__ import annotations

import numpy as np

from dynamic_programming_case_study import policy_iteration_with_history
from gridworld_case_study import GridWorldCaseStudyEnv, format_policy
from q_learning_case_study import q_learning_with_history


def print_dp_snapshots(dp_history):
    print("Dynamic Programming Snapshots")
    for snapshot in dp_history:
        print(
            f"\nIteration {snapshot['iteration']} "
            f"(stable={snapshot['stable']})"
        )
        print("V")
        print(snapshot["V"].reshape(3, 3))
        print("Greedy policy")
        print(format_policy(snapshot["policy"]))


def print_q_learning_snapshots(ql_snapshots):
    print("\nQ-Learning Snapshots")
    for snapshot in ql_snapshots:
        q_values = snapshot["Q"]
        print(
            f"\nEpisode {snapshot['episode']} "
            f"(epsilon={snapshot['epsilon']:.3f})"
        )
        print("max_a Q(s, a)")
        print(q_values.max(axis=1).reshape(3, 3))
        print("Greedy policy")
        print(format_policy(snapshot["policy"]))


def main():
    env = GridWorldCaseStudyEnv()

    dp_policy, dp_v, dp_q, dp_history = policy_iteration_with_history(env)
    ql_q, ql_snapshots = q_learning_with_history(env)
    ql_policy = ql_snapshots[-1]["policy"]
    ql_v = ql_q.max(axis=1)

    np.set_printoptions(precision=3, suppress=True)

    print_dp_snapshots(dp_history)
    print_q_learning_snapshots(ql_snapshots)

    print("\nFinal Comparison")
    print("\nDynamic Programming: V*")
    print(dp_v.reshape(3, 3))
    print("\nQ-Learning: max_a Q(s, a)")
    print(ql_v.reshape(3, 3))

    print("\nDynamic Programming Policy")
    print(format_policy(dp_policy))
    print("\nQ-Learning Policy")
    print(format_policy(ql_policy))

    print("\nPolicy match by state")
    print((dp_policy.argmax(axis=1) == ql_policy.argmax(axis=1)).reshape(3, 3))

    diff = np.abs(dp_q - ql_q)
    print("\nAbsolute Q-table difference |Q*_DP - Q_QL|")
    print(diff)
    print(f"\nMean absolute difference: {diff.mean():.4f}")
    print(f"Max absolute difference: {diff.max():.4f}")


if __name__ == "__main__":
    main()
