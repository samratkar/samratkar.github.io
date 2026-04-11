---
tags : [drl, rl]
title: "Chapter 3: Dynamic Programming"
category: "drl"
subcategory: "dynamic-programming"
---

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


### Source Code Dynamic Programming


<a href="https://github.com/samratkar/samratkar.github.io/blob/main/assets/drl/webinars/dp-qlearning/src/dynamic_programming_case_study.ipynb" target="_blank" rel="noopener noreferrer">
    Open for source code for Dynamic Programming case study
</a> 

### The gridboard game 

<a href="/assets/drl/webinars/dp-qlearning/src/dynamic_programming_game.html" target="_blank" rel="noopener noreferrer">
    Open the game in the a new tab full screen mode
</a> 


### Key Points - 
1. DP is a planning method that computes exact value functions from a known model.
2. DP uses the Bellman equations to iteratively compute values.
3. DP can compute both state values and action values, which are useful for different purposes.
4. DP is not a learning method in the sense of learning from experience, but it provides
the theoretical foundation for later learning methods that do learn from experience.
5. In DP, the $P(s', r \mid s, a)$ is constant, and it does not change. that is known as the model. In Q-learning, we do not have access to this model, and we learn from samples instead.
6. However, in DP we do learn and optimize on the $\pi (a \mid s)$, which is the policy. So, environment model is constant. But the action plicies change. However it is started with a static arbitrary policy which is then improved. That process is known as bootstrapping. 
7. In Q-learning, we also learn and optimize on the policy, but we do it indirectly by learning $Q(s,a)$ values that guide the policy. 
8. DP is : 
- model-based
- expectation-based
- sweep over all states and actions
- policy is udpated between interactions.