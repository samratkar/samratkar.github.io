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

> Q-network maps **state -> Q-values** for **all** actions, that are possible from **that state**.
The output depends on the number of actions and number of states. 
So for 5 input states, we get 5 rows, each row has 4 actions values (one per action).

1. If action space has 4 actions:
  - input one state tensor (state_dim,) -> output (4,) : 1 dimension vector
  - input batch of 5 states (5, state_dim) -> output(5,4) : 2 dimension matrix
2. NN's Output index corresponds to the action ID. 
  if the output of the network is [q0,q1,q2,q3]. That means : 
  - q0 = Q(s, action 0)
  - q1 = Q(s, action 1)
  - etc.  
3. `argmax` picks the idex of highest Q-value, and that index is the action we send to env.step(action).  


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
{% include_relative lunar_lander1_basic.py %}
{% endhighlight %}

![](./lunar_lander1_basic.py)

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
1. The learning is only from the last experience. So, consecutive updates are highly correlated.
2. Agent is forgetful.

### The Solution 
Experience Replay buffer - a memory that stores the agent's experiences at each time step, $e_t = (s_t, a_t, r_{t+1}, s_{t+1})$. During training, we sample mini-batches of experiences from the replay buffer to update the Q network. This breaks the correlation between consecutive updates and allows the agent to learn from a more diverse set of experiences.

## Improvising the training loop - Part 2 - Replay Buffering 

{% highlight python %}
{% include_relative lunar_lander2_replay_buffer.py %}
{% endhighlight %}

![](./lunar_lander2_replay_buffer.py)

### Dequeue data structure 

{% highlight python %}
{% include_relative dequeue.py %}
{% endhighlight %}

![](./dequeue.py)

### Replay Buffer implementation with dequeue 

{% highlight python %}
{% include_relative replay_buffer.py %}
{% endhighlight %}

![](./replay_buffer.py)

### Batch-wise processing 

1. The q-network gives all action scores for one state.
  Example for 4 actions: 
  ```python 
  q_values = [1.2, -0.4, 0.7, 2.0]
  ```

2. Suppose in that state, agent actually took action 2.

3. For training, you only need score of taken action: 

```python 
q_values[2] = 0.7
```

4. In batches, this happens for many rows at once.
  Example:                                                                            
  
  ```python
  q_values = 
  [                   
    [1.2, -0.4, 0.7, 2.0],   # sample 1
    [0.1,  0.3, 1.1, 0.5],   # sample 2
    [2.2,  1.9, 0.2, 0.0]    # sample 3
  ]
  
  actions = [2, 0, 1]  
  ```

5. We want:         
  - row1 pick col2 -> 0.7 
  - row2 pick col0 -> 0.1
  - row3 pick col1 -> 1.9

6. `gather()` does exactly this row-wise picking: 
```python 
  actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
  chosen_q = q_values.gather(1, actions_tensor)
  - action id = q-value index = before: shape (3,) -> [2,0,1]
  - action id = q-value index = after: shape (3,1) -> [[2],[0],[1]]
  - gather(dim=1, ...) needs this (batch, 1) form.
  chosen_q =
  [
    [0.7],
    [0.1],
    [1.9]
  ]
```
That is the value used in loss (predicted Q(s,a) vs target).