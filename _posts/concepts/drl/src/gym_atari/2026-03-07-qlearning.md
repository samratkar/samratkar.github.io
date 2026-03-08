---
tags : [drl, deep-queue, gymnasium]
title : "Deep Q Learning Notes"
category: dlr 
subcategory: "Deep Q Learning"
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

### A neural network to implement the Q function for Lunar Lander environment

```python 
# QNetwork(state_size, action_size). for Lunar Lander action_size = 4
# Action space - [0, 1, 2, 3] - [do nothing, fire left engine, fire main engine, fire right engine]
# State space - 8 dimensions - [x, y, x_dot, y_dot, angle, angular_velocity, left_leg_contact, right_leg_contact]
``` 

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

### Full algorithm with replay buffer

{% highlight python %}
{% include_relative lunar_lander2_replay_buffer.py %}
{% endhighlight %}

![](./lunar_lander2_replay_buffer.py)

## The complete DQN algorithm 
### The Problem 
1. enough exploration was not done. 
2. There were no targets for Q-values. 

### Epsilon Greediness - more exploration
Epsilon greediness lets the agent occasionally choose a random action over the highest value one. **Decayed greediness** can be followed to focus more on exploration early in training and more on exploitation later.

$\epsilon = end + (start - end) \cdot e^{-\frac{step}{decay}}$

![](/assets/drl/decayedepsilon.jpg)

This is implemented in the `select_action()` function - 
it requires 5 arguments - 
- `q_values`: The Q-values for all actions in the current state. This determines the optimal action. 
- `step` : the current step number 
- `start`, `end` and `decay` - thre parameters describing the epsilon decay.


```python
# slect action based on decayed epsilon greedy method
def select_action(q_values, step, start, end, decay):
    # calculate the threshold value for this step 
    epsilon = (end + (start-end)*math.exp(-step/decay))
    # draw a random number between 0 and 1
    sample = random.random()
    if sample < epsilon:
        # Return the random action index 
        return random.choice(range(len(q_values)))
    #Return the action index with the highest Q value
    return torch.argmax(q_values).item()
```

### Fixed Q value - more stable learning

The loss function is such that the target keeps changing with the network's own predictions. 
This is because - 
- Q-network is used in both q-value and TD Target calculation 
- this shifts both q-value and the target value. 
This can lead to instability. To address this, we use a **separate target network** that is a copy of the main Q-network but with frozen weights. The target network is updated periodically (e.g., every few episodes) with the weights of the main Q-network. This way, the target values are more stable during training.

![](/assets/drl/lossfun.jpg)

#### Target network

A target neural network predicts the target Q-values, and its weights are updated less frequently than the main network. This helps to stabilize training by providing a more consistent target for the loss function.

The online Q-network is updated every step, while the target network is updated every few episodes (or steps) by copying the weights from the online network.

#### Implementation 
1. **Initialize** the online and target networks with the same parameters.
```python 
# Initialize online and target networks with same initial parameters.
online_network = QNetwork(8, 4)
target_network = QNetwork(8,4)
target_network.load_state_dict(online_network.state_dict())
```
2. **Run gradient descent** on the online network every step, to determine the q-values for all the states in the batch. 

3. **Periodically not at every step, but once in a batch, update the target network** weights and biases with that of the **weighted average** of that of the online network. This way, the target network is updated less frequently, providing more stable targets for the loss function. 
A small value `tau` is a hyperparameter that controls the update rate of the target network. A common choice is `tau = 0.001`, which means that the target network is updated with 0.1% of the online network's weights and 99% of its own weights at each update step.

```python
def update_target_network(target_network, online_network, tau):
    target_net_state_dict = target_network.state_dict()
    online_net_state_dict = online_network.state_dict()
    for key in online_net_state_dict:
        target_net_state_dict[key] = (online_net_state_dict[key]*tau) + target_net_state_dict[key] * (1-tau)
    target_network.load_state_dict(target_net_state_dict)
    return None
```

4. **Training loop** - the complete DQN algorithm with replay buffer, epsilon greedy action selection and target network update. 

```python 
# 1. get the current state from environment 
state, info = env.reset()

# 2. get the q-values for all the current states from the online network 
q_values = online_network(state)

# 3. select action using epsilon greedy method
action = select_action(q_values, step, start, end, decay)

# 4. take action in the environment, get reward and next state
next_state, reward, done, info = env.step(action)

# 5. store the experience in replay buffer
replay_buffer.append((state, action, reward, next_state, done))

# 6. sample a batch of experiences from the replay buffer after replay buffer has enough samples
if len(replay_buffer) >= batch_size:
    batch = replay_buffer.sample(batch_size)

# 7. Identify the q_values of the current states and the actions taken in those states from the batch
q_values = online_network(states).gather(1, actions).squeeze(1)

# 8. compute the target Q-values using the target network. Don't compute gradients for the target network, as it is not being updated every step.
with torch.no_grad():
  # obtain the next state q_values across all columns in a given row
  next_state_q_values = target_network(next_states).amax(1)
  target_q_values = rewards + GAMMA * next_state_q_values * (1-dones)

# 9. compute the loss between the predicted Q-values from the online network and the target Q-values
loss = nn.MSELoss()(target_q_values, q_values)

# 10. perform a gradient descent step to update the online network's weights
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 11. periodically update the target network's weights with that of the online network using a weighted average
update_target_network(target_network, online_network, tau)
```

### Full implementation of the DQN algorithm with replay buffer, epsilon greedy action selection and target network update

{% highlight python %}
{% include_relative lunar_lander3_greedy_fixedQ.py %}
{% endhighlight %}
![](./lunar_lander3_greedy_fixedQ.py)

## Double DQN - to address overestimation bias in Q-learning
In standard DQN, the same network is used to select the action and to evaluate the action's value, which can lead to overestimation of Q-values. Double DQN addresses this by using the online network to select the action and the target network to evaluate the action's value. This helps to reduce overestimation bias and leads to more stable learning.