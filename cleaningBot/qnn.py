import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from visualize import *

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_shape, capacity):
        self.s = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.sp = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.capacity, self.count = capacity, 0

    def __len__(self):
        return self.count

    def add(self, s, a, r, sp, done):
        idx = self.count % self.capacity
        self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.done[idx] = s, a, r, sp, float(done)
        self.count += 1

    def sample(self, batch_size):
        max_idx = min(self.count, self.capacity)
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return (
            torch.from_numpy(self.s[idxs]).permute(0, 3, 1, 2).to(DEVICE),  # [B, 3, n, n]
            torch.from_numpy(self.a[idxs]).unsqueeze(1).to(DEVICE),
            torch.from_numpy(self.r[idxs]).unsqueeze(1).to(DEVICE),
            torch.from_numpy(self.sp[idxs]).permute(0, 3, 1, 2).to(DEVICE),  # [B, 3, n, n]
            torch.from_numpy(self.done[idxs]).unsqueeze(1).to(DEVICE),
        )


class QNet(nn.Module):
    def __init__(self, input_channels, n_actions, n):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)  # input: 3x20x20
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)  # input: 16x8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)  # input: 32x3x3

        self.fc1 = nn.Linear(64 * 1 * 1, 128)  # flatten size from conv3
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        # x: (batch, 3, 20, 20)
        x = F.relu(self.conv1(x))  # after conv1
        x = F.relu(self.conv2(x))  # after conv2
        x = F.relu(self.conv3(x))  # after conv3

        x = x.reshape(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNTrainer:
    def __init__(self, env, gamma=0.99, alpha=3e-4, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.3,
                 n_episodes=10000, max_steps=400, target_update_freq=20, batch_size=32, buffer_capacity=10000, warmup = 300):
        self.env, self.gamma, self.alpha = env, gamma, alpha
        self.epsilon, self.epsilon_decay, self.min_epsilon = epsilon, epsilon_decay, min_epsilon
        self.n_episodes, self.max_steps = n_episodes, max_steps
        self.target_update_freq, self.batch_size = target_update_freq, batch_size
        self.warmup = warmup

        self.n_actions = 4
        self.state_shape = env.state_dim  # (n, n, 3)
        n, _, c = self.state_shape[0], self.state_shape[1], self.state_shape[2]

        self.qnet = QNet(input_channels=c, n_actions=self.n_actions, n=n).to(DEVICE)
        self.q_target = QNet(input_channels=c, n_actions=self.n_actions, n=n).to(DEVICE)
        self.q_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(self.state_shape, buffer_capacity)

    def train(self):
        rewards = []
        bestReward = -1000000
        for ep in range(1, self.n_episodes + 1):
            s, total_reward = self.env.reset(), 0
            for stepCount in range(self.max_steps):
                if random.random() < self.epsilon:
                    a = random.randint(0, self.n_actions - 1)
                else:
                    with torch.no_grad():
                        state_t = torch.from_numpy(s).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # [1, 3, n, n]
                        qs = self.qnet(state_t)
                        a = int(torch.argmax(qs, dim=1).item())

                sp, r, done, _ = self.env.step(a, stepCount == self.max_steps-1)
                self.buffer.add(s, a, r, sp, done)
                total_reward += r
                s = sp

                if len(self.buffer) >= max(self.batch_size, self.warmup):
                    sb, ab, rb, spb, db = self.buffer.sample(self.batch_size)
                    q_pred = self.qnet(sb).gather(1, ab)
                    with torch.no_grad():
                        # Online net chooses the best action
                        next_actions = self.qnet(spb).argmax(1, keepdim=True)
                        # Target net evaluates it
                        q_next = self.q_target(spb).gather(1, next_actions)
                        target = rb + self.gamma * q_next * (1.0 - db)
                    loss = self.loss_fn(q_pred, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
                    self.optimizer.step()

                if done: break

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if ep % self.target_update_freq == 0:
                self.q_target.load_state_dict(self.qnet.state_dict())

            rewards.append(total_reward)
            print(f"Episode {ep} | Reward={total_reward:.2f} | Eps={self.epsilon:.3f}")

            if total_reward > bestReward and ep > 300:
                torch.save(self.qnet.state_dict(), f"weights.pth")
                bestReward = total_reward

        return rewards


class DQNAgent:
    def __init__(self, env):
        n, _, c = env.state_dim
        model = QNet(input_channels=c, n_actions=4, n=n).to(DEVICE)
        model.load_state_dict(torch.load("weights.pth"))
        model.eval()
        self.qnet = model

    def act(self, state):
        with torch.no_grad():
            state_t = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # [1, 3, n, n]
            qs = self.qnet(state_t)
        return int(torch.argmax(qs, dim=1).item())
