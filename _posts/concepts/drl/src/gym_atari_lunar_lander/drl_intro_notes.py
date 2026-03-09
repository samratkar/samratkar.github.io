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

env = gym.make("ALE/SpaceInvaders-V5")
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

# instantiate optimizer 
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

## the basic loop 
## iterate through episodes, each episode contains different states and different cumulative reward, defining a trajectory.
for episode in range (1000):
    state, info = env.reset()
    done = False
    ## building one trajectory, of all states in that. looping through all the states in a trajectory. 
    while not done:
        ## select an action using neural network. 
        action = select_action(network, state)
        ## observe the next state and reward from the env
        next_state, reward, terminated, tuncated, _ = (env.step(action))
        done = terminated or tuncated 
        ## calculate loss. agent creates its own training data based on their experience.
        loss = calculate_loss(network, state, action, next_state, reward, done)
        ## update the neural network based on the loss.
        # instantiate optimizer 
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state