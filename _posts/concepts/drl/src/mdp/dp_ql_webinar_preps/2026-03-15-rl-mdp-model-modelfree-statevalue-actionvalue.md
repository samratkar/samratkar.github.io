---
tags : [drl, mdp, dynamic-programming, q-learning, bellman, state-value, action-value]
title : "MPD - Model - Model Free - State Value - Action Value"
category: dlr
subcategory: "value-action-state-action"
layout : mermaid
---
# MPD - Model - Model Free - State Value - Action Value
## Environment and Agent & Their respective 2 conditional probabilities. 
### Conditional Probability distribution of next state and reward : Environment 
Environment is something that is not in the control of the agent. it is an outside factor. There is a **conditional probability of state transition** that is given by the environment. That is represented as - 

$$Environment = P(s',r|s,a)$$

Environment is defined by the **conditional probability**  of reaching to the next state $s'$ and getting a reward $r$ from the environment, given current state is $s$ and action taken is $a$. 

### Conditional probability distribution of action : Policy of the Agent
Agent is something that it can control. Action is controlled by the agent. It is depicted as a **conditional probability** known as **Policy** of the agent. It gives the probability of taking an action $a$ by the agent, given the current state is $s$. That is represented as -

$$Policy = \pi(a|s) $$

### Visual depiction of $\pi(a|s)$ and $P(s'|s,a)$

The two conditional probabilities act at different points in one RL step:

- $\pi(a|s)$ belongs to the **agent**: in state $s$, how likely is each action?
- $P(s'|s,a)$ belongs to the **environment**: after action $a$ is chosen in state $s$, how likely is each next state?

![](./pi_p_depiction.mermaid)

Read it left to right:

- first, the agent samples or selects an action using $\pi(a|s)$
- then, given that chosen action, the environment produces the next state according to $P(s'|s,a)$

So one interaction step is:

$$
s \xrightarrow{\pi(a|s)} a \xrightarrow{P(s'|s,a)} s'
$$

If reward is included explicitly, the fuller environment model is:

$$
P(s', r \mid s, a)
$$

and $P(s'|s,a)$ is the marginal next-state distribution obtained from it.

## State Value and Action Value

State value and action value are two different ways to represent the expected return in reinforcement learning.

| State value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Action Value or Q=value                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| State value, denoted as $V(s)$, represents the expected return starting from state $s$ and following a particular policy $\pi$ thereafter. It is defined as:$$V^\pi(s) = \mathbb{E}[ return \mid start \;  in \; s, \; then \; follow \; policy \; π \; till \; the \; goal]$$                                                                                                                                                                                                                                                                                              | State value, denoted as $V(s)$, represents the expected return starting from state $s$ and following a particular policy $\pi$ thereafter. It is defined as:$$V^\pi(s) = \mathbb{E}[ return \mid start \;  in \; s, \; then \; follow \; policy \; π \; till \; the \; goal]$$                                                                                                                                                                                                         |
| $V_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) \big[ r + \gamma V_\pi(s') \big]$<br>first term is the conditional probability sweep for all agent's action policy - $\pi(a \mid s)$ conditional probability of taking an action, given the agent is in a given state s. This sweep is done for all action probabilities leading to the expected value.<br>second term is a sweep for all such conditional probabilities of the environment's state transition $P(s',r \mid s, a)$ for the a chosen in the 1st term.<br>This is a function of only state $s$ | $Q_\pi(s, a)= \sum_{s', r} P(s', r \mid s, a) \Big[ r + \gamma \sum_{a'} \pi(a' \mid s') , Q_\pi(s', a') \Big]$<br>for an action selected as $a$ in the current state, the 1st term sweeps across all the sates in the environment's state transition probability $P(s',r \mid s)$ and finds the expected reward. <br>Then, it recursively does what $V(s)$ was doing, i.e. go over the sweep of all action probabilities and multiply withe state transition probabilities recursively. |
| $V(s)$ averages over the action probabilities $\pi(a \mid s)$ and then over state transition probabilities $P(s',r \mid s, a)$                                                                                                                                                                                                                                                                                                                                                                                                                                             | $Q(s, a)$ fixes the action (no averaging over $\pi$ at that step), but still depends on the state transition probabilities $P(s',r \mid s,a)$ for next state onwards                                                                                                                                                                                                                                                                                                                  |
| $V_\pi(s) = \mathbb{E}_\pi[ R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s ]$<br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | $Q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| $ \left[V_\pi(s) = \sum_a \pi(a\mid s) , Q_\pi(s,a) \right] $<br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| $V_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s]$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | <br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |


!!! info "Key Insight"
    **Both the state value and action value store the same value except that the action value is recorded after an agent takes an action at its current state.**

## Definition of Return

In reinforcement learning, the **return** at time step \(t\), denoted \(G_t\), is defined as the (discounted) sum of future rewards:

\[
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
\]

- \(R_{t+1}\) is the reward received **after** taking the action at time \(t\).
- \(\gamma \in [0,1)\) is the discount factor.
- There is **no \(R_t\) term** in \(G_t\); by definition, the return from time \(t\) looks only at rewards from the **next** time step onward.

Using this, the state-value function under a policy \(\pi\) can be written as an expectation over returns:

\[
V_\pi(s) = \mathbb{E}_\pi[ G_t \mid S_t = s ].
\]

By expanding the recursion \(G_t = R_{t+1} + \gamma G_{t+1}\), we get the Bellman expectation form:

\[
\begin{align}
V_\pi(s)
  &= \mathbb{E}_\pi[ G_t \mid S_t = s ] \\
  &= \mathbb{E}_\pi[ R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s ] \\
  &= \sum_a \pi(a \mid s)
     \sum_{s', r} P(s', r \mid s, a) \big[ r + \gamma V_\pi(s') \big].
\end{align}
\]

## State Value and Policy
- State value `V(s)` is defined for a single state `s`.
- It represents the **expected** total reward **from** that state to the **goal** when following a particular **policy**.
- The value of a state depends on the entire future trajectory of states and rewards under the policy, but it is attached to the starting state only.

## Episode Start and State Value
- The state value depends on where the episode starts.
- For a trajectory `s0 → s1 → s2 → s3 → s4`:
  - `V(s0)` is the expected sum of rewards from `s0` onward.
  - `V(s1)` is the expected sum of rewards from `s1` onward.
- The episode start determines which state value is relevant.

## Validity of State Values
- For a given policy, there is a valid state value for every state in the Markov Decision Process (MDP).
- The episode boundaries (start and end) determine which state values are used or estimated.

## Definition of an Episode
- An episode typically starts at an initial state and ends at a terminal state (goal, failure, or timeout).
- The full sequence from start to terminal state is called a **trajectory or episode.**
- Value functions can be defined as if **starting at any** state along the trajectory for analysis. But always ending with the goal state. 

## Expected Rewards and Sampling

### Definition level (theory)
- The state value `V(s)` is the **expected return**, which is a probability-weighted sum of rewards and next-state values (Bellman equation).

If we have the full MDP model (all transition probabilities and rewards), we can write something like the Bellman equation, to compute the state value $V(s)$:

```math
\begin{align}
V_\pi(s)
  &= \mathbb{E}_\pi[ G_t \mid S_t = s ] \\
  &= \mathbb{E}_\pi[ R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s ] \\
  &= \sum_a \pi(a \mid s) 
     \sum_{s', r} P(s', r \mid s, a) \big[ r + \gamma V_\pi(s') \big].
\end{align}
```

$$V_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} P(s', r | s, a) [ r + γ V(s') ]$$

>**outer sum over a** : weighted average based on probability (Expected value) according to the policy π.
**inner sum over (s',r)** : weighted average based on the transition probabilities of the MDP, over all transitions and rewards.

in layman's term : 

$$V_\pi(s) = \mathbb{E}[ return | start \;  in \; s, \; then \; follow \; policy \; π \; till \; the \; goal]$$

$$V_\pi(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s]$$

Here the expectation is explicit: probabilities × rewards/next-values, summed.

- In practice, the agent samples trajectories and averages actual returns to estimate `V(s)`, as an approximation to the expected value, as the state transition probabilities might not be known to the agent.
- Sampling does not multiply rewards by transition probabilities explicitly because the environment's randomness and averaging over episodes implicitly do this.

When we know the full MDP (transition probabilities and rewards **$P(s', r | s, a)$.**) and use the Bellman equation directly to determine the value function, that family of methods is called **dynamic programming** - policy evaluation, policy iteration, value iteration. 

### Sample level (practice) 
In many RL settings, the agent doesn’t know transition probabilities and rewards - **$P(s', r | s, a)$.** 

So instead of computing the expectation analytically, it does the following - 

- Interacts with the environment and gets sampled trajectories: **`s0, a0, r1`** → **`s1, a1, r2`**, …
- For each visit to state s, it computes the actual return from that point:
**$$G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + …$$** 
- Then it averages these returns over many episodes to **approximate** V(s).

So at the sample level, we don’t multiply each reward by transition probabilities because We usually don’t know those probabilities. The law of large numbers tells us that averaging many sampled returns converges to the same expectation that the probability-weighted sum would give.

### Rule of the Thumb

**Theory:** “V(s) is the average outcome over all possible futures, weighted by their probabilities.”

**Practice:** “I’ll run the process many times from s, record the actual returns, and take the empirical average. That implicitly does the probability-weighting for me.”

## Numerical Example of Sampling vs Definition

### Scenario
- State `s` with two actions `a1` and `a2`.
- Action `a1`: reward 5 with probability 0.8, reward 0 with probability 0.2.
- Action `a2`: reward 10 with probability 0.5, reward 0 with probability 0.5.

### Definition Method
- `V(s)` for `a1` = 0.8 * 5 + 0.2 * 0 = 4
- `V(s)` for `a2` = 0.5 * 10 + 0.5 * 0 = 5

### Sample Method
- Sample episodes for each action and record rewards.
- Average returns converge to expected values as samples increase.

### Conclusion
- Sampling method estimates converge to the definition method values by the law of large numbers.

## RL Methods on the “Definition vs Sample” Axis
### Model Based
#### 1. Dynamic Programming (DP) – Definition / Model-Based
- Assumes **full knowledge of the MDP environment**: conditional transition probabilities and rewards.
- Uses the **Bellman expectation/optimality equations** directly.
- Computes values via **sweeps over the state space**:
  - Policy evaluation
  - Policy iteration
  - Value iteration
- Conceptually: *“I know all probabilities, so I can compute expectations analytically.”*
### Model Free
#### 1. Monte Carlo (MC) – Sample-Based, No Bootstrapping 
- **Model-free**: does **not** need transition probabilities. But it has the policiy distribution $\pi(a|s)$ to follow.
- Uses **complete episodes** sampled from interaction with the environment.
- For each state, averages **actual returns** observed after visiting that state.
- No bootstrapping: updates use **only sampled returns**, not current value estimates.
- Conceptually: *“I don’t know probabilities; I’ll approximate expectations by averaging many full-episode samples.”*

#### 2. Temporal-Difference (TD) Methods – Sample-Based, With Bootstrapping
Model-free like MC, but uses **bootstrapping** (updates from other estimates):

#### 2.1 SARSA (On-Policy TD Control)
- Learns **action-value function** \( Q(s, a) \).
- Update uses the **actual next action** taken under the current policy:
  - Target: \( r + \gamma Q(s', a') \).
- On-policy: learns about the policy it is actually following (e.g., ε-greedy).

#### 2.2 Q-learning (SARSAMAX, Off-Policy TD Control)
- Also learns **Q(s, a)**.
- Update uses the **max over next actions**:
  - Target: \( r + \gamma \max_{a'} Q(s', a') \).
- Off-policy: learns the **greedy** policy while possibly behaving ε-greedily for exploration.

### The Big Picture
- **Dynamic Programming**: definition/analytic level with a **known model**.
- **Monte Carlo + TD (SARSA, Q-learning)**: **sample level**, learning from experience without knowing the model.
  - MC: sample-based, no bootstrapping.
  - TD (SARSA, Q-learning): sample-based **with** bootstrapping.

<div class="mermaid">
flowchart TB
  A["Bellman Equations (Theory)<br/>Definition level"]
  A --> B{"Do we know the full MDP?<br/>(transitions & rewards)"}

  B -->|Yes| C["Dynamic Programming (DP)<br/>Model-based"]
  B -->|No| D["Model-free RL<br/>Sample level"]

    C --> C1[Policy Evaluation]
    C --> C2[Policy Iteration]
    C --> C3[Value Iteration]

    D --> E["Monte Carlo (MC)"]
    D --> F["Temporal-Difference (TD)"]

    E:::mc
    F:::td

    F --> G["SARSA<br/>(on-policy TD control)"]
    F --> H["Q-learning<br/>(off-policy TD control)"]

    classDef mc fill:#e0f7fa,stroke:#00838f,stroke-width:1px;
    classDef td fill:#fff3e0,stroke:#ef6c00,stroke-width:1px;
</div>

## Action Value 

### State value V(s) (under a policy π):

$V^\pi(s) = \mathbb{E}[ return | start \;  in \; s, \; then \; follow \; policy \; π \; till \; the \; goal]$

```math
\begin{align}
V_\pi(s)
  &= \mathbb{E}_\pi[ G_t \mid S_t = s ] \\
  &= \mathbb{E}_\pi[ R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s ] \\
  &= \sum_a \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) \big[ r + \gamma V_\pi(s') \big].
\end{align}
```

>This expectation already weight-averages over all possible actions according to the policy π and over the resulting transitions. (the first summation)

## Action value Q(s, a) (under a policy π):

$Q^\pi(s, a) = \mathbb{E}[ return | start \; in \; s, \; take \; action \; \textbf{a} \; once, \; then \; follow \; π ]$
Here we fix the first action to be a, so we look at the distribution P(s', r | s, a) induced by that specific action.


So:

$V^\pi(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s]$
$Q^\pi(s, a) = \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a]$

>So, **State Value** can be written as follows in-terms of **action value** 
$$ \left[V_\pi(s) = \sum_a \pi(a\mid s) , Q_\pi(s,a) \right] $$

This means:

- At state s, the policy π may choose different actions with probabilities π(a|s).
- Vπ(s) is the expected return, averaging over all actions the policy might take and their resulting stochastic transitions.
- So yes: you can think of Vπ(s) as “the action choice is handled automatically by the policy; I just take the expectation over that action distribution and the environment’s randomness.”

What **Qπ(s, a)** is doing - Action value - 

$$Q_\pi(s,a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]$$

This means:
- We force the first action to be ***a*** at state ***s***.
- Then we still average over: The stochastic transitions $P(s', r | s, a)$ (so it’s not deterministic), and The future actions chosen by π from s' onward.

So: Qπ(s,a) is not deterministic; it’s still an expectation over all possible next states, rewards, and future actions, but conditioned on the fact that the first action was a.
Vπ(s) then averages those Qπ(s,a) values over actions using π(a|s).

You can summarize it like this:

Qπ(s,a): “If I commit to action a now in s, what is my expected return (given the environment’s randomness and my policy afterward)?”
Vπ(s): “If I just follow π in s (letting it randomly pick actions), what is my expected return?”

The ***‘universe’*** is all possible first actions and their futures under $\pi$ (that’s what $V_\pi(s)$ averages over).
Each $Q_\pi(s,a)$ is like zooming into the subset where the first action is fixed to $a$.

| State value                                                                                                                        | Action Value or Q=value                                                                                                                                               |
|:-----------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Expected value of reward in a particular state.                                                                                    | Expected value of reward in a particular state, given an action a is taken.                                                                                           |
| $V_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) \big[ r + \gamma V_\pi(s') \big]$                                 | $Q_\pi(s, a)= \sum_{s', r} P(s', r \mid s, a) \Big[ r + \gamma \sum_{a'} \pi(a' \mid s') , Q_\pi(s', a') \Big]$                                                       |
| $V_\pi(s) = \mathbb{E}_\pi[ R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s ]$                                                        | $Q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$                                                                                                                  |
| $ \left[V_\pi(s) = \sum_a \pi(a\mid s) , Q_\pi(s,a) \right] $                                                                      |                                                                                                                                                                       |
| $V_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s]$                                                                                        |                                                                                                                                                                       |
| $V(s)$ averages over the action probabilities $\pi(a \mid s)$ and then over state transition probabilities $P(s',r \mid s, a)$  | $Q(s, a)$ fixes the action (no averaging over $π$ at that step), but still depends on the state transition probabilities P(s',r \mid s,a) for next state onwards  |
