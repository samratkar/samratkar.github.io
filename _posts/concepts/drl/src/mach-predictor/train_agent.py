import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import json
import argparse
from aircraft_env import AircraftEnv

# Hyperparameters
DEFAULT_CONFIG = "configs/approach_expanded_fuel_model_golden.json"

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh() # Actions are in [-1, 1]
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) # Learnable std deviation
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self):
        raise NotImplementedError

    def get_action_and_value(self, state, action=None, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Add batch dimension if missing
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        if action is None:
            action = mean if deterministic else dist.sample()
        
        action_log_probs = dist.log_prob(action).sum(axis=-1)
        dist_entropy = dist.entropy().sum(axis=-1)
        value = self.critic(state)
        
        return action, action_log_probs, dist_entropy, value

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train(config_path=None):
    if config_path is None:
        config_path = DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None
    cfg = load_config(config_path) if config_path else {}

    LEARNING_RATE = cfg.get("learning_rate", 2e-4)
    GAMMA = cfg.get("gamma", 0.99)
    EPS_CLIP = cfg.get("eps_clip", 0.2)
    K_EPOCHS = cfg.get("k_epochs", 4)
    BATCH_SIZE = cfg.get("batch_size", 128)
    TOTAL_TIMESTEPS = cfg.get("total_timesteps", 250000)
    HIDDEN_SIZE = cfg.get("hidden_size", 64)
    PRETRAIN_STEPS = cfg.get("pretrain_steps", 2000)
    PRETRAIN_BATCH = cfg.get("pretrain_batch", 256)
    SHAPING_STRENGTH = cfg.get("reward_shaping_strength", 0.5)
    CURRICULUM_STEPS = cfg.get("curriculum_steps", 80000)
    QAR_DATA_PATH = cfg.get("qar_data_path", "data/qar_737800_cruise.csv")
    FUEL_FLOW_SCALE = cfg.get("fuel_flow_scale", 1.0)
    FUEL_FLOW_BIAS = cfg.get("fuel_flow_bias", 0.0)
    FUEL_FPN_QUAD = cfg.get("fuel_fpn_quad")
    FUEL_MODEL_PATH = cfg.get("fuel_model_path")
    FUEL_MODEL_SCALER_PATH = cfg.get("fuel_model_scaler_path")
    # Create Environment
    # We use the synthetic data for initialization
    data_path = QAR_DATA_PATH if os.path.exists(QAR_DATA_PATH) else "data/Tail_X1.csv"
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Running with random initialization.")
        env = AircraftEnv(
            reward_shaping_strength=SHAPING_STRENGTH,
            phase_mode="cruise_only",
            fuel_flow_scale=FUEL_FLOW_SCALE,
            fuel_flow_bias=FUEL_FLOW_BIAS,
            fuel_fpn_quad=FUEL_FPN_QUAD,
            fuel_model_path=FUEL_MODEL_PATH if FUEL_MODEL_PATH and os.path.exists(FUEL_MODEL_PATH) else None,
            fuel_model_scaler_path=FUEL_MODEL_SCALER_PATH if FUEL_MODEL_SCALER_PATH and os.path.exists(FUEL_MODEL_SCALER_PATH) else None,
        )
    else:
        env = AircraftEnv(
            data_path=data_path,
            reward_shaping_strength=SHAPING_STRENGTH,
            phase_mode="cruise_only",
            fuel_flow_scale=FUEL_FLOW_SCALE,
            fuel_flow_bias=FUEL_FLOW_BIAS,
            fuel_fpn_quad=FUEL_FPN_QUAD,
            fuel_model_path=FUEL_MODEL_PATH if FUEL_MODEL_PATH and os.path.exists(FUEL_MODEL_PATH) else None,
            fuel_model_scaler_path=FUEL_MODEL_SCALER_PATH if FUEL_MODEL_SCALER_PATH and os.path.exists(FUEL_MODEL_SCALER_PATH) else None,
        )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(state_dim, action_dim, hidden_size=HIDDEN_SIZE)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")

    # 0. Oracle pretraining (behavior cloning on coarse grid search)
    print("Pretraining actor on oracle Mach...")
    for _ in range(PRETRAIN_STEPS):
        states = []
        targets = []
        for _ in range(PRETRAIN_BATCH):
            s, _ = env.reset()
            # Convert oracle Mach to normalized action
            oracle_mach = env._oracle_mach()
            target_action = (oracle_mach - 0.78) / 0.08
            states.append(torch.FloatTensor(s))
            targets.append(torch.FloatTensor([target_action]))
        states = torch.stack(states)
        targets = torch.stack(targets)
        pred, _, _, _ = policy.get_action_and_value(states, deterministic=True)
        loss = ((pred - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
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
        # Curriculum: start with cruise-only, then full phases
        if global_step == CURRICULUM_STEPS:
            env.phase_mode = "mixed"

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
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    args = parser.parse_args()
    train(args.config)
