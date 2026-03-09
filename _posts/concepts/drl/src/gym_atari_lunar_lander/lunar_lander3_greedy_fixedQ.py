"""
Greedy Epsilon and fix Q-value 
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import random 
import math

from q_network import QNetwork
from replay_buffer import ReplayBuffer


GAMMA = 0.99
LR = 1e-4
NUM_EPISODES = 100
batch_size = 64
replay_buffer = ReplayBuffer(capacity=10000)
tau = 0.001

q_network = QNetwork(8, 4)
optimizer = torch.optim.Adam(q_network.parameters(), lr=LR)
criterion = nn.MSELoss()
online_network = QNetwork(8, 4)
target_network = QNetwork(8,4)
target_network.load_state_dict(online_network.state_dict())

def update_target_network(target_network, online_network, tau):
    target_net_state_dict = target_network.state_dict()
    online_net_state_dict = online_network.state_dict()
    for key in online_net_state_dict:
        target_net_state_dict[key] = (online_net_state_dict[key]*tau) + target_net_state_dict[key] * (1-tau)
    target_network.load_state_dict(target_net_state_dict)
    return None


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



def to_tensor(state):
    return torch.tensor(state, dtype=torch.float32)

def describe_episode(episode, reward, episode_reward, step, terminated, truncated, total_steps):
    if truncated:
        status = "Timeout"
    elif episode_reward >= 200:
        status = "Solved"
    elif episode_reward >= 100:
        status = "Landed"
    elif episode_reward >= 0:
        status = "Improving"
    elif episode_reward >= -100:
        status = "Stabilizing"
    else:
        status = "Crashed"

    print(
        f"| Episode {episode + 1:4d} | Duration: {step:4d} steps | Reward: {episode_reward:10.2f} | {status:<12} | Total Steps: {total_steps:6d} |"
    )


def plot_rewards(episode_rewards):
    episodes = list(range(1, len(episode_rewards) + 1))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trend_filename = f"lunar_lander3_rewards_trend_{timestamp}.png"
    hist_filename = f"lunar_lander3_rewards_histogram_{timestamp}.png"
    x = np.asarray(episodes, dtype=np.float32)
    y = np.asarray(episode_rewards, dtype=np.float32)

    # Linear trend line: y = m*x + b
    m, b = np.polyfit(x, y, 1)
    trend = m * x + b

    # Line plot with trend line
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards, color="tab:blue", linewidth=1.8, label="Episode Reward")
    plt.plot(episodes, trend, color="tab:red", linewidth=2.0, linestyle="--", label="Trend")
    plt.title("LunarLander Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(trend_filename, dpi=150)

    # Histogram of achieved rewards
    plt.figure(figsize=(10, 5))
    plt.hist(episode_rewards, bins=15, color="tab:green", edgecolor="black", alpha=0.75)
    plt.title("Histogram of Episode Rewards")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(hist_filename, dpi=150)
    print(f"Saved plots: {trend_filename}, {hist_filename}")

    plt.show()


env = gym.make("LunarLander-v3", render_mode="human")
all_episode_rewards = []
total_steps = 0
for episode in range(NUM_EPISODES):
    state, info = env.reset()
    done = False
    step = 0
    episode_reward = 0.0

    while not done:
        step += 1
        total_steps += 1
        # invokes the online_network and passes the inputs states, and it returns q_values for all the states - 1 row for each state. 
        q_values = online_network(state)
        # select the action with epsilon greediness
        action = select_action(q_values, total_steps, start=0.9, end=0.05, decay=1000)
        # apply the chosen action to the environment for one timestep, till done. 
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # store the latest experience in the replay buffer. 
        replay_buffer.push(state, action, reward, next_state, done)

        # sample a batch of 64 experiences from the replay buffer 
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            q_values = online_network(states).gather(1, actions).squeeze(1)
            # don't update the weights of the target network during backward propagation. gradients are not tracked.
            with torch.no_grad():
                # obtain the next state q_values across all columns in a given row
                next_state_q_values = target_network(next_states).amax(1)
                target_q_values = rewards + GAMMA * next_state_q_values * (1-dones)
            loss = nn.MSELoss()(target_q_values, q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_target_network(target_network, online_network, tau=0.005)
        state = next_state
        episode_reward += reward

    all_episode_rewards.append(episode_reward)
    describe_episode(episode, reward, episode_reward, step, terminated, truncated, total_steps)

env.close()
plot_rewards(all_episode_rewards)
