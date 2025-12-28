---
layout: mermaid
title: Introduction to Reinforcement learning using Gymnasium
description: "Gymnasium is a toolkit for developing and comparing reinforcement learning algorithms. It provides a wide variety of environments to test and train RL agents."
tags: [gymnasium, reinforcement-learning, rl, python]
---

![](../../../../images/rl-intro.png)
![](../../../../images/rl-fwk.png)

```python
env = create_environment()
state = env.get_initial_state()
for i in range (n_iterations):
   action = choose_action(state)
   state, reward = env.execute (action)
   update_knowledge(state, action, reward)
```
Tasks - 
1. episodic - tasks segmented into episodes. episode has a beginning and an end. 
2. continuous 

# Key points
1. RL is based on reward for desirable behaviors and punishments for undesirable ones.
2. RL is based on interaction **between an agent and an environment**, to achieve a s**pecific goal.**
3. ![](/images/rl/rl-not-rl.png)