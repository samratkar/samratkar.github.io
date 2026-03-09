import argparse
import collections
import random
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers import AtariPreprocessing, FrameStack


Transition = collections.namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        sample = random.sample(self.buffer, batch_size)
        states = torch.tensor(
            np.array([np.array(entry.state, copy=False) for entry in sample], dtype=np.float32),
            device=device,
        )
        next_states = torch.tensor(
            np.array([np.array(entry.next_state, copy=False) for entry in sample], dtype=np.float32),
            device=device,
        )
        actions = torch.tensor([entry.action for entry in sample], dtype=torch.int64, device=device)
        rewards = torch.tensor([entry.reward for entry in sample], dtype=torch.float32, device=device)
        dones = torch.tensor([float(entry.done) for entry in sample], dtype=torch.float32, device=device)
        return states / 255.0, actions, rewards, next_states / 255.0, dones

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, frames: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(frames, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainingConfig:
    total_frames: int = 200_000
    replay_size: int = 50_000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-4
    update_frequency: int = 4
    target_update: int = 1_000
    learning_starts: int = 5_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.02
    epsilon_decay: int = 100_000
    eval_every: int = 10_000
    eval_episodes: int = 3
    save_path: Optional[str] = "space_invaders_dqn.pt"
    load_path: Optional[str] = None
    eval_only: bool = False


def make_env() -> gym.Env:
    base_env = gym.make("ALE/SpaceInvaders-V5")
    env = AtariPreprocessing(
        base_env,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_newaxis=True,
        scale_obs=False,
    )
    env = FrameStack(env, 4)
    return env


def unwrap_reset(result):
    return result[0] if isinstance(result, tuple) else result


def unwrap_step(result):
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, terminated or truncated, info
    observation, reward, done, info = result
    return observation, reward, done, info


def epsilon_by_frame(frame_idx: int, start: float, end: float, decay: int) -> float:
    progress = min(frame_idx / decay, 1.0)
    return start + (end - start) * progress


def select_action(state, network, epsilon, action_space_n, device):
    if random.random() < epsilon:
        return random.randrange(action_space_n)
    tensor = torch.tensor(np.array(state, copy=False), dtype=torch.float32, device=device) / 255.0
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        return int(network(tensor).argmax(dim=1).item())


def optimize_model(policy_net, target_net, buffer, batch_size, gamma, optimizer, device):
    if len(buffer) < batch_size:
        return None
    states, actions, rewards, next_states, dones = buffer.sample(batch_size, device)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
    expected_q = rewards + gamma * next_q_values * (1 - dones)
    loss = F.smooth_l1_loss(q_values, expected_q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()


def evaluate_policy(policy_net, device, episodes):
    policy_net.eval()
    eval_env = make_env()
    returns = []
    for _ in range(episodes):
        state = np.array(unwrap_reset(eval_env.reset()), copy=False)
        done = False
        episode_reward = 0.0
        while not done:
            action = select_action(state, policy_net, 0.0, eval_env.action_space.n, device)
            next_state, reward, done, _ = unwrap_step(eval_env.step(action))
            state = np.array(next_state, copy=False)
            episode_reward += reward
        returns.append(episode_reward)
    eval_env.close()
    policy_net.train()
    return float(np.mean(returns)), float(np.std(returns))


def train(config: TrainingConfig, device: torch.device):
    env = make_env()
    action_space_n = env.action_space.n
    policy_net = DQN(4, action_space_n).to(device)
    target_net = DQN(4, action_space_n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.replay_size)

    if config.load_path:
        policy_net.load_state_dict(torch.load(config.load_path, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())

    frame_idx = 0
    episode_reward = 0.0
    episode_rewards = []
    losses = []

    state = np.array(unwrap_reset(env.reset()), copy=False)

    while frame_idx < config.total_frames:
        epsilon = epsilon_by_frame(frame_idx, config.epsilon_start, config.epsilon_final, config.epsilon_decay)
        action = select_action(state, policy_net, epsilon, action_space_n, device)
        next_state, reward, done, _ = unwrap_step(env.step(action))
        next_state = np.array(next_state, copy=False)

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            state = np.array(unwrap_reset(env.reset()), copy=False)
            episode_rewards.append(episode_reward)
            episode_reward = 0.0

        if (
            len(buffer) >= config.batch_size
            and frame_idx > config.learning_starts
            and frame_idx % config.update_frequency == 0
        ):
            loss = optimize_model(policy_net, target_net, buffer, config.batch_size, config.gamma, optimizer, device)
            if loss is not None:
                losses.append(loss)

        if frame_idx % config.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if config.eval_every and frame_idx > 0 and frame_idx % config.eval_every == 0:
            eval_mean, eval_std = evaluate_policy(policy_net, device, config.eval_episodes)
            avg_loss = float(np.mean(losses[-100:])) if losses else 0.0
            last_reward = episode_rewards[-1] if episode_rewards else 0.0
            print(
                f"{frame_idx:08d} frames | eps={epsilon:.3f} | reward={last_reward:.1f} |"
                f" loss={avg_loss:.4f} | eval={eval_mean:.1f}+/-{eval_std:.1f}"
            )
            if config.save_path:
                torch.save(policy_net.state_dict(), config.save_path)

        frame_idx += 1

    env.close()
    print("Training finished.")
    if config.save_path:
        torch.save(policy_net.state_dict(), config.save_path)
    return policy_net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Space Invaders")
    parser.add_argument("--total-frames", type=int, default=200_000, help="Number of environment frames to gather")
    parser.add_argument("--replay-size", type=int, default=50_000, help="Capacity of the replay buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for gradient updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--update-frequency", type=int, default=4, help="Steps between optimizer updates")
    parser.add_argument("--target-update", type=int, default=1_000, help="Steps between target network syncs")
    parser.add_argument("--learning-starts", type=int, default=5_000, help="Warm-up frames before learning")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-final", type=float, default=0.02, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=int, default=100_000, help="Frames over which epsilon anneals")
    parser.add_argument("--eval-every", type=int, default=10_000, help="Frames between evaluation runs")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Episodes to run during evaluation")
    parser.add_argument("--save-path", type=str, default="space_invaders_dqn.pt", help="Path to persist the policy weights")
    parser.add_argument("--load-path", type=str, default=None, help="Optional checkpoint to warm-start from")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate a saved policy")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        total_frames=args.total_frames,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        update_frequency=args.update_frequency,
        target_update=args.target_update,
        learning_starts=args.learning_starts,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay=args.epsilon_decay,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        save_path=args.save_path,
        load_path=args.load_path,
        eval_only=args.eval_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.eval_only:
        assert config.load_path, "--eval-only requires --load-path"
        temp_env = make_env()
        net = DQN(4, temp_env.action_space.n).to(device)
        temp_env.close()
        net.load_state_dict(torch.load(config.load_path, map_location=device))
        mean_return, std_return = evaluate_policy(net, device, config.eval_episodes)
        print(f"Evaluation only | mean_return={mean_return:.1f}+/-{std_return:.1f}")
        return

    train(config, device)


if __name__ == "__main__":
    main()
