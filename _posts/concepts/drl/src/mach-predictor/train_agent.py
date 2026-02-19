import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
from aircraft_env import AircraftEnv

# Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
TOTAL_TIMESTEPS = 100000
HIDDEN_SIZE = 64

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Tanh() # Actions are in [-1, 1]
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) # Learnable std deviation
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self):
        raise NotImplementedError

    def get_action_and_value(self, state, action=None):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Add batch dimension if missing
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        action_log_probs = dist.log_prob(action).sum(axis=-1)
        dist_entropy = dist.entropy().sum(axis=-1)
        value = self.critic(state)
        
        return action, action_log_probs, dist_entropy, value

def train():
    # Create Environment
    # We use the synthetic data for initialization
    data_path = "data/Tail_X1.csv"
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Running with random initialization.")
        env = AircraftEnv()
    else:
        env = AircraftEnv(data_path=data_path)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    
    # Simple PPO Rollout Buffer
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    
    global_step = 0
    ep_reward = 0
    state, _ = env.reset()
    
    for step in range(TOTAL_TIMESTEPS):
        # 1. Collect Data
        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(state)
            
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        
        states.append(torch.FloatTensor(state))
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        
        state = next_state
        ep_reward += reward
        global_step += 1
        
        if done:
            state, _ = env.reset()
            # print(f"Episode Reward: {ep_reward:.2f}") # Too noisy for step-by-step
            ep_reward = 0

        # 2. Update Policy (PPO) every batch
        if len(states) >= BATCH_SIZE:
             # Bootstrap value if not done
            with torch.no_grad():
                _, _, _, next_value = policy.get_action_and_value(state)
                # Compute returns / advantages
                # Simplified GAE or just discounted returns for short horizon
                # Since horizon is 1 (bandit), Return = Reward.
                # But let's write it generically for multi-step
                
                returns = []
                gae = 0
                for i in reversed(range(len(rewards))):
                    # For 1-step bandit, next_val is irrelevant if done=True, which it always is
                    # So Delta = r + 0 - v
                    mask = 1.0 - dones[i]
                    delta = rewards[i] + GAMMA * next_value.item() * mask - values[i].item()
                    gae = delta + GAMMA * 0.95 * mask * gae
                    ret = gae + values[i].item()
                    returns.insert(0, ret)
                    next_value = values[i] # Approximate
            
            # Flatten
            b_states = torch.stack(states)
            b_actions = torch.stack(actions)
            b_log_probs = torch.stack(log_probs)
            b_returns = torch.FloatTensor(returns)
            b_values = torch.stack(values).squeeze()
            b_advantages = b_returns - b_values
            
            # Optimize
            for _ in range(K_EPOCHS):
                _, new_log_probs, entropy, new_values = policy.get_action_and_value(b_states, b_actions)
                ratio = (new_log_probs - b_log_probs).exp()
                
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * b_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * ((new_values.squeeze() - b_returns)**2).mean()
                entropy_loss = -0.01 * entropy.mean()
                
                loss = actor_loss + critic_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Clear buffer
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            
            if global_step % 10000 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")

    # Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(policy.state_dict(), "models/tail_policy.pt")
    print("Training finished. Model saved to models/tail_policy.pt")

if __name__ == "__main__":
    train()
