"""
loop through all episodes 
    loop through steps in each episode
        at each step choose an action, 
        calculate the loss,
        update the network
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    def forward(self, x):
        return self.linear(x)

# instantiate the network. nn works as a policy to map states to actions. 
# input is 8 : set of one state representation. here 8 inputs constitute one input state representation. 
# output is number of actions. 4 in this case.
network = Network(8, 4)

env = gym.make("LunarLander-v2")
# Run ten episodes
for episode in range(10):
    state, info = env.reset()
    done = False    
    # Run through steps until done
    while not done:
        action = select_action(network, state)        
        # Take the action
        next_state, reward, terminated, truncated, _ = (env.step(action))
        done = terminated or truncated        
        loss = calculate_loss(network, state, action, next_state, reward, done)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        # Update the state
        state = next_state 
    print(f"Episode {episode} complete.")