---
tags : [drl, big-pic, gymnasium]
title : "Deep Reinforcement Learning (DRL) - Big Picture"
category: dlr 
subcategory: "big-pic"
layout : mermaid
---

# Part 1 - Tabular Solution Methods 
1. value functions are represented as arrays or tables. 
2. single state Multi Armed Bandit (MAB)
3. Markov Decision Process (MDP) - states, actions, transition probabilities, rewards. Bellman and value functions.
4. 3 fundamental classes of solving MDP - 
    - Dynamic Programming (DP) - requires a complete and accurate model of the environment's dynamics (transition probabilities and rewards). It uses this model to compute optimal policies through methods like value iteration and policy iteration.
    - Monte Carlo (MC) methods - do not require a model of the environment. They learn from experience by sampling episodes of interaction with the environment and using the observed rewards to estimate value functions
    - Temporal Difference (TD) learning - also do not require a model of the environment. They learn from experience by updating value estimates based on the observed rewards and the estimated value of the next state, without waiting for the end of an episode.
5. MAB 
{% include_relative mab/2026-03-09-mab.md %}
