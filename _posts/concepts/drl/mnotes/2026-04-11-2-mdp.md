---
tags : [drl, rl]
title: "Chapter 2: Markov Decision Processes"
category: "drl"
subcategory: "mdp"
---

## Markov Decision Processes (MDP)

A Markov Decision Process (MDP) is a mathematical framework used to model decision-making problems where outcomes are partly random and partly under the control of a decision-maker. An MDP is defined by the following components:
1. **States ($S$)**: A set of all possible states in the environment.
2. **Actions ($A$)**: A set of all possible actions that the agent can take.
3. **Transition Probability ($P(s'|s,a)$)**: The probability of transitioning to state $s'$ given that the agent is in state $s$ and takes action $a$.
4. **Reward Function ($R(r|s,a)$)**: The probability of receiving reward $r$ given that the agent is in state $s$ and takes action $a$.
5. **Discount Factor ($\gamma$)**: A factor between 0 and 1 that represents the importance of future rewards compared to immediate rewards.

## The Objective of an MDP
The goal of an agent in an MDP is to learn a policy ($\pi(a|s)$) that maximizes the expected cumulative reward over time. The Bellman equation is a fundamental recursive relationship that describes the value of a state or action in terms of the expected rewards and the values of subsequent states or actions. Solving an MDP typically involves finding an optimal policy that maximizes the expected cumulative reward, which can be achieved through various algorithms such as dynamic programming, Monte Carlo methods, and temporal difference learning.

## Settings

In dynamic programming, we assume the finite MDP is fully known:

$$
\mathcal{M} = \left(\mathcal{S}, \mathcal{A}, \pi(a \mid s), p(s', r \mid s, a), \gamma\right)
$$

In plain English, this means:

- the environment is modeled as an MDP $\mathcal{M}$
- $\mathcal{S}$ is the set of all states
- $\mathcal{A}$ is the set of all actions
- $\pi(a \mid s)$ is the policy that maps states to actions
- $p(s', r \mid s, a)$ is the conditional probability of getting next state $s'$ and reward $r$, given current state $s$ and action $a$
- $\gamma \in [0,1)$ is the discount factor used to discount future rewards


Once the MDP model $\mathcal{M}$ is known, dynamic programming uses it to compute long-term return estimates. These are the state-value function $V_\pi(s)$ and the action-value function $Q_\pi(s,a)$. 

For policy evaluation, the state-value function is:

$$
V^{\pi}(s) = \sum_a \pi(a \mid s)\sum_{s',r} p(s', r \mid s, a)\left[r + \gamma V^{\pi}(s')\right]
$$

For action values:

$$
Q^{\pi}(s,a) = \sum_{s',r} P(s', r \mid s, a)\left[r + \gamma \sum_{a'} \pi(a' \mid s') Q^{\pi}(s', a')\right]
$$

$$Q^{\pi}(s,a) = \sum_{s',r} P(s',r \mid s,a)\left[r + \gamma V^{\pi}(s')\right]$$

