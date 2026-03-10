---
tags : [drl, mdp, dynamic-programming, q-learning, gymnasium]
title : "Case Study: Dynamic Programming vs Q-Learning"
category: dlr
subcategory: "mdp"
layout : mermaid
---

# Case Study: Dynamic Programming vs Q-Learning

This note presents two small case studies:

1. Dynamic Programming on a known grid world
2. Q-Learning on the same grid world without using the model

The goal is to show:
- how both methods try to solve the same decision problem
- why their update mechanisms are different
- when one is more appropriate than the other

## Common Environment

Consider a `3 x 3` grid world:

```text
+-----+-----+-----+
| S0  | S1  | S2  |
+-----+-----+-----+
| S3  | S4  | S5  |
+-----+-----+-----+
| S6  | S7  | S8  |
+-----+-----+-----+
```

Assumptions:
- start state: `S0`
- goal state: `S8`
- actions: `Up`, `Right`, `Down`, `Left`
- reward `+10` for reaching `S8`
- reward `-1` for every non-terminal move
- episode ends at `S8`

State layout:

```text
S0 S1 S2
S3 S4 S5
S6 S7 S8
```

Action mapping:

```python
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
```

---

## Case Study 1: Dynamic Programming

### Setting

In dynamic programming, we assume the environment model is fully known.

That means for every state-action pair we know:
- the next possible state
- the reward
- the transition probability

In a tabular Gym-style setting, that is represented by:

```python
env.P[s][a]
```

### Intuition

Dynamic programming computes values by applying Bellman updates repeatedly.

In this grid world:
- if moving `Right` from `S7` reaches `S8`, that action should become highly valued
- then states that can reach `S7` should also gain value
- eventually value information propagates backward through the whole grid

### Diagram

```mermaid
flowchart LR
    S0["S0"] --> S1["S1"]
    S1 --> S2["S2"]
    S2 --> S5["S5"]
    S5 --> S8["S8 Goal"]

    S0 --> S3["S3"]
    S3 --> S6["S6"]
    S6 --> S7["S7"]
    S7 --> S8
```

Dynamic programming uses the full model to evaluate all possible actions and transitions at every state.

### Example Policy Iteration Code

```python
import numpy as np


def policy_evaluation(env, policy, gamma=0.9, theta=1e-8):
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    while True:
        delta = 0.0
        for s in range(num_states):
            old_v = V[s]
            new_v = 0.0

            for a, action_prob in enumerate(policy[s]):
                if action_prob == 0:
                    continue

                for prob, next_state, reward, done in env.P[s][a]:
                    new_v += action_prob * prob * (
                        reward + gamma * (1 - done) * V[next_state]
                    )

            V[s] = new_v
            delta = max(delta, abs(old_v - new_v))

        if delta < theta:
            break

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
    Q = compute_q_from_v(env, V, gamma)
    num_states, num_actions = Q.shape
    policy = np.zeros((num_states, num_actions))

    for s in range(num_states):
        best_action = np.argmax(Q[s])
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
            break

        policy = new_policy

    return policy, V, Q
```

### Example Outcome

A likely optimal policy for the grid might look like:

```text
+---------+---------+---------+
| S0  ->  | S1  ->  | S2  v   |
+---------+---------+---------+
| S3  ->  | S4  ->  | S5  v   |
+---------+---------+---------+
| S6  ->  | S7  ->  | S8 Goal |
+---------+---------+---------+
```

This means:
- from the top row, move right until the last column
- then move down toward the goal

### What DP is doing

Dynamic programming is effectively asking:

- if I know the exact transition probabilities and rewards,
- what is the exact best value of each state,
- and therefore what is the exact best action at each state?

So DP performs **full expected backups**.

---

## Case Study 2: Q-Learning

### Setting

Now assume we do **not** know the model.

So we do not know:
- `env.P[s][a]`
- transition probabilities
- exact reward structure in table form

Instead, the agent only interacts with the environment and observes:

```text
(state, action, reward, next_state)
```

### Intuition

Q-learning starts with a zero Q-table and improves it from experience.

At first:
- the agent explores
- values are mostly guesses

Over many episodes:
- transitions leading toward the goal receive better value
- bad actions get lower value
- the Q-table gradually approximates the optimal action-value function

### Q-Learning Update

```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)\right]
```

### Diagram

```mermaid
flowchart LR
    A["Current state s"] --> B["Choose action a"]
    B --> C["Observe reward r and next state s'"]
    C --> D["Compute target: r + gamma max_a' Q(s',a')"]
    D --> E["Update Q(s,a)"]
```

### Q-Learning Code

```python
import numpy as np
import random


def epsilon_greedy_action(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(Q.shape[1])
    return np.argmax(Q[state])


def q_learning(env, num_episodes=5000, alpha=0.1, gamma=0.9, epsilon=1.0,
               epsilon_decay=0.995, epsilon_min=0.01):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_action(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward + gamma * (1 - done) * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q


def greedy_policy_from_q(Q):
    num_states, num_actions = Q.shape
    policy = np.zeros((num_states, num_actions))

    for s in range(num_states):
        best_action = np.argmax(Q[s])
        policy[s, best_action] = 1.0

    return policy
```

### Example Learning Story

Suppose the agent is in `S7` and takes `Right`.

If that reaches the goal `S8` with reward `+10`, then:

```text
Q(S7, Right) becomes large
```

Then later, if the agent is in `S6` and moving `Right` often leads to `S7`, then:

```text
Q(S6, Right) also increases
```

So just like DP, value information propagates backward, but here it happens from sampled experience instead of from the known model.

### Example Learned Policy

After enough episodes, the learned greedy policy may become:

```text
+---------+---------+---------+
| S0  ->  | S1  ->  | S2  v   |
+---------+---------+---------+
| S3  ->  | S4  ->  | S5  v   |
+---------+---------+---------+
| S6  ->  | S7  ->  | S8 Goal |
+---------+---------+---------+
```

So the final policy may match the dynamic programming solution, even though the learning process is completely different.

---

## Side-by-Side Comparison

### Core Difference

Dynamic Programming:
- knows the model
- computes exact expected updates

Q-Learning:
- does not know the model
- learns from sampled experience

### Backup Comparison

Dynamic programming backup:

```math
Q^*(s,a) = \sum_{s',r} p(s',r \mid s,a)\left[r + \gamma \max_{a'} Q^*(s',a')\right]
```

Q-learning backup:

```math
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
```

Interpretation:
- DP averages over **all possible next outcomes**
- Q-learning uses **one observed sample**

### Diagram

```mermaid
flowchart TD
    A["Dynamic Programming"]
    B["Known model"]
    C["Enumerate all next outcomes"]
    D["Expected Bellman backup"]
    E["Exact update"]

    F["Q-Learning"]
    G["Unknown model"]
    H["Observe one transition sample"]
    I["Sampled Bellman backup"]
    J["Incremental update"]

    A --> B --> C --> D --> E
    F --> G --> H --> I --> J
```

### Comparison Table

| Aspect | Dynamic Programming | Q-Learning |
|---|---|---|
| Model needed? | Yes | No |
| Uses `env.P[s][a]`? | Yes | No |
| Update type | Expected backup | Sampled backup |
| Learns from | Full model | Experience |
| Requires episodes? | Not necessarily | Usually trained over episodes |
| Exploration needed? | No | Yes |
| Typical setting | Known tabular MDP | Unknown environment |
| Converges to optimal? | Yes, under standard assumptions | Yes, under standard assumptions and sufficient exploration |

---

## Why Both Matter

Dynamic programming is important because:
- it gives the clean mathematical foundation
- it shows what the exact Bellman solution looks like
- it explains policy evaluation, policy improvement, policy iteration, and value iteration

Q-learning is important because:
- in real problems the model is often unknown
- we still want to learn optimal behavior
- Q-learning shows how Bellman optimality can be learned from data

So the relationship is:

```text
Dynamic Programming:
  exact Bellman updates with known model

Q-Learning:
  sample-based Bellman updates without known model
```

---

## Final Takeaway

Both methods try to answer the same question:

- what is the best action to take in each state?

But they solve it differently:

- Dynamic Programming solves it from the model
- Q-Learning solves it from experience

So if the model is known, dynamic programming is the cleanest exact method.
If the model is unknown, Q-learning is a practical model-free alternative.

---

## Key Insights in Sutton-Style Framing

The following ideas are closely aligned with the way Sutton and Barto motivate these methods.

### Key insights behind Dynamic Programming

1. **Dynamic programming assumes a complete model of the environment**

DP requires knowledge of:
- state transitions
- rewards

This is why DP is mainly a planning method rather than a pure learning method.

2. **DP is built directly on the Bellman equations**

The Bellman expectation and Bellman optimality equations provide recursive definitions of value.
DP turns those recursive equations into iterative algorithms.

3. **Policy evaluation and policy improvement are the central decomposition**

A major insight is that control can be broken into two alternating steps:
- evaluate the current policy
- improve the policy using the current values

This leads naturally to policy iteration.

4. **Value iteration compresses evaluation and improvement together**

Instead of fully evaluating a policy before improving it, value iteration performs partial evaluation and greedy improvement in the same update.

5. **DP establishes the conceptual foundation for later RL methods**

Even when a model is not available, the structure of DP remains important because later methods such as TD learning and Q-learning can be understood as approximations to DP-style Bellman backups.

### Key insights behind Q-Learning

1. **Q-learning learns action values directly from experience**

Rather than learning a model first, Q-learning directly estimates:
$$
Q^*(s,a)
$$

This is a major shift from planning with a model to learning from interaction.

2. **Q-learning is a sample-based version of Bellman optimality updates**

Instead of averaging over all possible next outcomes, Q-learning updates from one sampled transition at a time.

So it can be seen as replacing:
- full expected backup

with:
- sampled backup

3. **Q-learning is off-policy**

This is one of the most important conceptual insights.

The agent may behave using an exploratory policy, such as epsilon-greedy, but the update target still uses:
$$
\max_{a'} Q(s',a')
$$

So it learns about the greedy policy while behaving differently during learning.

4. **The learned Q-values are enough for control**

Once the Q-table is learned, there is no need to separately learn a model or a state-value function.
The policy can be obtained directly by choosing:
$$
\arg\max_a Q(s,a)
$$

5. **Q-learning brings optimal control into the model-free setting**

This is the core reason it is so influential.
It keeps the optimality idea from dynamic programming, but removes the need for a known transition model.

### Shared insight across both methods

Both dynamic programming and Q-learning are driven by the same high-level idea:

- good decisions today depend on reward today plus good decisions tomorrow

In Sutton-style RL, this is the unifying role of Bellman recursion.

So:
- DP uses Bellman recursion with a known model
- Q-learning uses Bellman recursion with sampled experience

### Short summary

```text
Dynamic Programming:
  planning with a known model
  exact Bellman backups
  policy evaluation + policy improvement

Q-Learning:
  learning without a model
  sampled Bellman optimality backups
  direct estimation of optimal action values
```
