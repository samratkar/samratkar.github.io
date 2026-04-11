---
tags : [drl, mdp, gymnasium, fdp]
title : "FDP workshop"
category: dlr 
subcategory: "rl_foundations"
layout : mermaid
professor : "Prof. Vimal S P"
---

# RL foundations 
- input : interaction with environment
- goal : learn a policy (mapping states to actions), that maximizes some utility function. 
    - policy can be deterministic \pi : S \to A 
    - policy can be stochastic \pi(a|s) = P(a|s)
    - good policy - when action is beneficial in long run
- value function : Q(s,a) :  chances that we will win is higher ==> higher value function. **cumulative rewards in long term** 

## Difference Between Reward and Value

In reinforcement learning:

- Reward is the immediate feedback from the environment after an action at time \(t\), usually written as \(r_t\).
- Value is the expected long-term return (future cumulative rewards), usually from a state or state-action pair.

Common forms:

- State value:
  \[
  V^\pi(s)=\mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k r_{t+k+1}\mid s_t=s\right]
  \]
- Action value (Q-value):
  \[
  Q^\pi(s,a)=\mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k r_{t+k+1}\mid s_t=s,a_t=a\right]
  \]

So: reward is a one-step signal, while value is the predicted total future reward.

- agent - anything and everything that has absolute control. 
- environment - everything else.
- $\pi(a|s)$ - policy, probability of taking action a in state s.
- environment - P(s|a) - gives the next state and reward. 
- terminal state - win, lose, draw. for tic tac toc
- table is the **value function** - state to value. 
- exploration and exploitation - interleave. - is heart of action selection. 

