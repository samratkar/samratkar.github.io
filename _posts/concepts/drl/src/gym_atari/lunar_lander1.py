"""
Simple LunarLander DQN-style training loop.
"""

import gymnasium as gym
import torch
import torch.nn as nn

from q_network import QNetwork

GAMMA = 0.99
LR = 1e-4
NUM_EPISODES = 10


model = QNetwork(8, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()


def to_tensor(state):
    return torch.tensor(state, dtype=torch.float32)


def select_action(net, state_tensor):
    q_values = net(state_tensor)
    return torch.argmax(q_values).item()


def calculate_loss(net, state, action, next_state, reward, done):
    state_t = to_tensor(state)
    next_state_t = to_tensor(next_state)

    q_values = net(state_t)
    current_q = q_values[action]

    with torch.no_grad():
        next_q = net(next_state_t).max()
        target_q = reward + GAMMA * next_q * (1 - int(done))

    return criterion(current_q, target_q)


env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action = select_action(model, to_tensor(state))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        loss = calculate_loss(model, state, action, next_state, reward, done)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        episode_reward += reward

    print(f"Episode {episode + 1}: reward={episode_reward:.2f}")

env.close()

