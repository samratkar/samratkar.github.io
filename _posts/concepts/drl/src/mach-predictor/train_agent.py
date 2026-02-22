import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import json
import argparse
import time
import shutil
from datetime import datetime
from pathlib import Path
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
    
    train_start = time.time()
    print("Starting training...")

    # 0. Oracle pretraining (behavior cloning on coarse grid search)
    pretrain_start = time.time()
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
    
    pretrain_elapsed = time.time() - pretrain_start
    print(f"Pretraining done in {pretrain_elapsed:.1f}s ({pretrain_elapsed/60:.1f} min)")

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
                elapsed = time.time() - train_start
                print(f"Step {global_step}, Loss: {loss.item():.4f}, Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save Model
    total_elapsed = time.time() - train_start
    os.makedirs("models", exist_ok=True)
    torch.save(policy.state_dict(), "models/tail_policy.pt")
    print("Training finished. Model saved to models/tail_policy.pt")
    print(f"Total training time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # --- Timestamped snapshot ---
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = Path(f"models/saved_{run_ts}")
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Copy weights
    shutil.copy("models/tail_policy.pt", snap_dir / "tail_policy.pt")
    for fname in ("fuel_model.pt", "fuel_model_scaler.json"):
        src = Path("models") / fname
        if src.exists():
            shutil.copy(src, snap_dir / fname)

    # Copy training config
    if config_path and Path(config_path).exists():
        shutil.copy(config_path, snap_dir / Path(config_path).name)

    # Write README
    fuel_model_r2   = "0.6938"
    fuel_model_mae  = "1.18 kg/NM"
    fuel_model_rmse = "1.88 kg/NM"

    loss_rows = "\n".join(
        f"| {s:>9,} | {l:.4f} | {e:.1f}s ({e/60:.1f} min) |"
        for s, l, e in loss_log
    )

    readme = f"""# Model Snapshot — {run_ts.replace("_", " ")[:4]}-{run_ts[4:6]}-{run_ts[6:8]} {run_ts[9:11]}:{run_ts[11:13]}:{run_ts[13:15]}

Auto-generated by `train_agent.py` at end of training run.

---

## Files

| File | Description |
|---|---|
| `tail_policy.pt` | PPO actor-critic policy weights |
| `fuel_model.pt` | Neural fuel-per-NM model weights (copied from parent) |
| `fuel_model_scaler.json` | Feature normalization stats (copied from parent) |
| `{Path(config_path).name if config_path else "N/A"}` | Training config used for this run |

---

## Performance Results

| Metric | Value |
|---|---|
| Evaluation rows | 20,000 (run `evaluate_qar.py` after training to populate) |
| Baseline fuel/NM | — |
| Policy fuel/NM | — |
| Improvement | — |

> Re-run `python evaluate_qar.py` pointing at this snapshot to get final numbers.

---

## Training Timing

| Phase | Duration |
|---|---|
| Oracle pretraining ({PRETRAIN_STEPS:,} steps × {PRETRAIN_BATCH} batch) | {pretrain_elapsed:.1f}s ({pretrain_elapsed/60:.1f} min) |
| PPO training ({TOTAL_TIMESTEPS:,} steps) | {total_elapsed - pretrain_elapsed:.1f}s ({(total_elapsed - pretrain_elapsed)/60:.1f} min) |
| **Total** | **{total_elapsed:.1f}s ({total_elapsed/60:.1f} min)** |

---

## Training Loss Progression

| Step | PPO Loss | Elapsed |
|---|---|---|
{loss_rows}

---

## PPO Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Total timesteps | {TOTAL_TIMESTEPS:,} |
| Batch size | {BATCH_SIZE} |
| K epochs per update | {K_EPOCHS} |
| Learning rate | {LEARNING_RATE} (Adam) |
| Discount γ | {GAMMA} |
| GAE λ | 0.95 |
| PPO clip ε | {EPS_CLIP} |
| Entropy coefficient | 0.01 |
| Hidden size | {HIDDEN_SIZE} |
| Activation | Tanh |
| Oracle pretrain steps | {PRETRAIN_STEPS:,} |
| Oracle pretrain batch | {PRETRAIN_BATCH} |
| Reward shaping strength | {SHAPING_STRENGTH} |
| Curriculum steps | {CURRICULUM_STEPS:,} (cruise-only → mixed) |
| Training data | `{QAR_DATA_PATH}` |
| Reward signal | Neural fuel model |

---

## PPO Policy Architecture (`tail_policy.pt`)

```
Actor:  Linear({state_dim}→{HIDDEN_SIZE}, Tanh) → Linear({HIDDEN_SIZE}→{HIDDEN_SIZE}, Tanh) → Linear({HIDDEN_SIZE}→{action_dim}, Tanh)
Critic: Linear({state_dim}→{HIDDEN_SIZE}, Tanh) → Linear({HIDDEN_SIZE}→{HIDDEN_SIZE}, Tanh) → Linear({HIDDEN_SIZE}→1)
log_std: learnable scalar [1, {action_dim}]
```

| Layer | Shape | Parameters |
|---|---|---|
| actor.0.weight | [{HIDDEN_SIZE}, {state_dim}] | {HIDDEN_SIZE * state_dim:,} |
| actor.0.bias | [{HIDDEN_SIZE}] | {HIDDEN_SIZE} |
| actor.2.weight | [{HIDDEN_SIZE}, {HIDDEN_SIZE}] | {HIDDEN_SIZE * HIDDEN_SIZE:,} |
| actor.2.bias | [{HIDDEN_SIZE}] | {HIDDEN_SIZE} |
| actor.4.weight | [{action_dim}, {HIDDEN_SIZE}] | {action_dim * HIDDEN_SIZE:,} |
| actor.4.bias | [{action_dim}] | {action_dim} |
| critic.0.weight | [{HIDDEN_SIZE}, {state_dim}] | {HIDDEN_SIZE * state_dim:,} |
| critic.0.bias | [{HIDDEN_SIZE}] | {HIDDEN_SIZE} |
| critic.2.weight | [{HIDDEN_SIZE}, {HIDDEN_SIZE}] | {HIDDEN_SIZE * HIDDEN_SIZE:,} |
| critic.2.bias | [{HIDDEN_SIZE}] | {HIDDEN_SIZE} |
| critic.4.weight | [1, {HIDDEN_SIZE}] | {HIDDEN_SIZE:,} |
| critic.4.bias | [1] | 1 |
| log_std | [1, {action_dim}] | {action_dim} |
| **Total** | | **{2*(HIDDEN_SIZE*state_dim + HIDDEN_SIZE + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE) + action_dim*HIDDEN_SIZE + action_dim + HIDDEN_SIZE + 1 + action_dim:,}** |

### Input (21 features, normalized)

altitude, grossWeight, TAT, CAS, tempDevC, windKts, phase, targetAltitude,
turbulence, regime, angleOfAttackVoted, horizontalStabilizerPosition,
totalFuelWeight, trackAngleTrue, fmcMach, latitude, longitude,
GMTHours, Day, Month, YEAR

### Output

`a ∈ [-1, 1]`  →  `Mach = 0.78 + a × 0.08`  →  `[0.70, 0.86]`

---

## Neural Fuel Model (`fuel_model.pt`)

```
Linear(17→64, ReLU) → Linear(64→64, ReLU) → Linear(64→1)
Target: fuel_per_nm normalized (y_mean=11.9788, y_std=3.3369 kg/NM)
```

| Metric | Value |
|---|---|
| R² | {fuel_model_r2} |
| MAE | {fuel_model_mae} |
| RMSE | {fuel_model_rmse} |
| Total parameters | 5,377 |
"""

    (snap_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Snapshot saved to {snap_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    args = parser.parse_args()
    train(args.config)
