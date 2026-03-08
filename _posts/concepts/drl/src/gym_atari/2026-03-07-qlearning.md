---
tags : [drl, deep-queue, gymnasium]
title : "Deep Q Learning Notes"
category: dlr 
subcategory: Deep Q Learning
layout : mermaid
---

# Deep Q learning 
## What is Q? 
 
`Action value function` **$Q(s,a)$** associating a value (reward) to any combination of state $s_t$ and action $a_t$.

![](/assets/drl/q_s_a.jpg)

## Recursive definition of Q

$Q(s_t,a_t)$ can be written as a recursive formula called the `Bellman equation`, expressing the **Q value** in the current state in terms of the Q values of the next states: 

![](/assets/drl/bellman-deterministic-q.jpg)

The **update rule** for Q learning - 

![](/assets/drl/q_learning_update_rule.jpg)

## Q Network
### Mapping states to action values

![](/assets/drl/q_nw.jpg)

`Bellman optimality equation` (for $Q^{*}$):

$$
Q^{*}(s_t,a_t) = \mathbb{E}\!\left[ r_t + \gamma \max_{a'} Q^{*}(s_{t+1},a') \;\middle|\; s_t=s,\; a_t=a \right]
$$

### A neural network to implement the Q function

{% highlight python %}
{% include_relative q_network.py %}
{% endhighlight %}

![](./q_network.py)

## Training the neural network with lunar lander environment - Part 1

![](/assets/drl/lossfun.jpg)

{% highlight python %}
{% include_relative lunar_lander1.py %}
{% endhighlight %}

![](./lunar_lander1.py)

### Output 
![](/assets/drl/output1.jpg)

```text
Episode 1: reward=-431.60
Episode 2: reward=-501.77
Episode 3: reward=-210.85
Episode 4: reward=-408.80
Episode 5: reward=-266.54
Episode 6: reward=23.38
Episode 7: reward=-368.02
Episode 8: reward=-95.81
Episode 9: reward=-398.78
Episode 10: reward=-329.06
```
### The Problem
The lunar lander crashes. Because - 
1. The agent starts with an untrained Q-network, so its Q-values are essentially random.
2. The policy is greedy (argmax) from the start, so it repeatedly picks whatever random action currently looks best.
3. There is no exploration strategy (no epsilon-greedy), so it does not try enough alternative actions to discover safer behavior.                                                                        
4. LunarLander needs coordinated action sequences; random/poor early choices quickly lead to bad trajectories and crashes.                                                                            
5. Training updates are noisy because they are done step-by-step on highly correlated samples (no replay buffer)
6. There is no target network, so the learning target moves every step, making Q-learning unstable.
7. Very short training (10 episodes) is far from enough for this task; early episodes are expected to be mostly crashes.
8. No reward/gradient stabilization (e.g., clipping) can further increase unstable updates in early training.  

## Improvising the training loop - Part 2
