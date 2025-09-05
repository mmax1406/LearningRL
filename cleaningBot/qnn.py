import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_dim, capacity):
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.sp = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.capacity, self.count = capacity, 0

    def __len__(self):
        return self.count

    def add(self, s, a, r, sp, done):
        idx = self.count % self.capacity
        self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.done[idx] = s.reshape(-1), a, r, sp.reshape(-1), float(done)
        self.count += 1

    def sample(self, batch_size):
        max_idx = min(self.count, self.capacity)
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return (
            torch.from_numpy(self.s[idxs]).to(DEVICE),
            torch.from_numpy(self.a[idxs]).unsqueeze(1).to(DEVICE),
            torch.from_numpy(self.r[idxs]).unsqueeze(1).to(DEVICE),
            torch.from_numpy(self.sp[idxs]).to(DEVICE),
            torch.from_numpy(self.done[idxs]).unsqueeze(1).to(DEVICE),
        )

class QNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class DQNTrainer:
    def __init__(self, env, gamma=0.95, alpha=1e-3, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1,
                 n_episodes=300, max_steps=50, target_update_freq=20, batch_size=32, buffer_capacity=10000):
        self.env, self.gamma, self.alpha = env, gamma, alpha
        self.epsilon, self.epsilon_decay, self.min_epsilon = epsilon, epsilon_decay, min_epsilon
        self.n_episodes, self.max_steps = n_episodes, max_steps
        self.target_update_freq, self.batch_size = target_update_freq, batch_size

        self.n_actions, self.state_dim = 4, env.state_dim
        self.qnet = QNet(self.state_dim, self.n_actions).to(DEVICE)
        self.q_target = QNet(self.state_dim, self.n_actions).to(DEVICE)
        self.q_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(self.state_dim, buffer_capacity)

    def train(self):
        rewards = []
        bestReward = 0
        for ep in range(1, self.n_episodes + 1):
            s, total_reward = self.env.reset(), 0
            for stepCount in range(self.max_steps):
                if random.random() < self.epsilon:
                    a = random.randint(0, self.n_actions - 1)
                else:
                    with torch.no_grad():
                        qs = self.qnet(torch.from_numpy(s).unsqueeze(0).to(DEVICE))
                        a = int(torch.argmax(qs, dim=1).item())

                sp, r, done, _ = self.env.step(a, stepCount == self.max_steps-1)
                self.buffer.add(s, a, r, sp, done)
                total_reward += r

                if len(self.buffer) >= self.batch_size:
                    sb, ab, rb, spb, db = self.buffer.sample(self.batch_size)
                    q_pred = self.qnet(sb).gather(1, ab)
                    with torch.no_grad():
                        q_next = self.q_target(spb).max(1, keepdim=True)[0]
                        target = rb + self.gamma * q_next * (1.0 - db)
                    loss = self.loss_fn(q_pred, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
                    self.optimizer.step()

                s = sp
                if done: break

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if ep % self.target_update_freq == 0:
                self.q_target.load_state_dict(self.qnet.state_dict())

            rewards.append(total_reward)

            print(f"Episode {ep} | Reward={total_reward:.2f} | Eps={self.epsilon:.3f}")

            # --- save weights ---
            if total_reward > bestReward:
                torch.save(self.qnet.state_dict(), f"weights.pth")
                bestReward = total_reward

        return rewards

class DQNAgent:
    def __init__(self):
        model = QNet(402, 4).to(DEVICE)
        model.load_state_dict(torch.load("weights.pth"))
        model.eval()  # disable dropout/batchnorm if any
        self.qnet = model

    def act(self, state):
        state = state.reshape(-1)  # flatten just in case
        with torch.no_grad():
            qs = self.qnet(torch.from_numpy(state).unsqueeze(0).to(DEVICE))
        return int(torch.argmax(qs, dim=1).item())