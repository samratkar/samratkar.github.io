---
tags : [drl, monte-carlo]
title : "Monte Carlo (MC) Methods"
category: dlr
subcategory: "monte-carlo"
layout : mermaid
---

## Key points 
1. In case of monte-carlo methods, we estimate the value function based on the average return of multiple episodes. We can use this method to estimate the value function for a given policy, which is known as policy evaluation.
2. The key idea behind monte-carlo methods is to use the actual returns obtained from episodes to update our estimates of the value function. This is in contrast to other methods, such as temporal-difference learning, which use bootstrapping to update estimates based on other estimates.
3. Monte-carlo methods can be used for both policy evaluation and policy improvement. In policy evaluation, we estimate the value function for a given policy, while in policy improvement, we use the estimated value function to improve the policy.
4. One of the main advantages of monte-carlo methods is that they can be used in environments with unknown dynamics, as they do not require a model of the environment. However, they can be computationally expensive, as they require multiple episodes to obtain accurate estimates of the value function.
5. The system is model-free. That means the conditional probability of state transition and rewards is not know - $P(s', r | s, a)$ is not known. We can only sample from the environment to get the next state and reward. This is in contrast to model-based methods, where we have a model of the environment that allows us to compute the next state and reward given the current state and action. However the other conditional probability of the policy is known - $\pi(a | s)$ is known. We can use this to sample actions from the policy given a state. This is an important distinction, as it allows us to use monte-carlo methods for policy evaluation and improvement without needing a model of the environment.
6. Even if the policy says, "in state s, always take action a", the next state and reward can be random. So, **different episodes can have different paths** under the same policy, as the enironment's state transition and reward functions can be stochastic. This is a key aspect of monte-carlo methods, as it allows us to estimate the value function based on the average return of multiple episodes, which can capture the variability in the environment's dynamics.

## The Process of Monte Carlo Methods

<div class="mermaid">
flowchart TD
    A[Start] --> B[Initialize arbitrary policy π]
    B --> C["Initialize Q(s,a) (e.g., zeros)"]

    C --> D[Repeat for many iterations]

    D --> E["Generate an episode<br/>by following current policy π<br/>(with exploration)"]

    E --> F["For each state-action (s,a)<br/>in the episode,<br/>compute return G from that time step"]

    F --> G["Update Q(s,a)<br/>using average of returns<br/>observed for (s,a)"]

    G --> H["Improve policy π:<br/>for each state s,<br/>choose action with highest Q(s,a)<br/>(e.g., ε-greedy)"]

    H --> D
</div>

