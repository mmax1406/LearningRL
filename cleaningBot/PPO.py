import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Actor-Critic Network (Conv for 3xNxN input)
# =====================
class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_dim, n=20):
        super().__init__()
        # Shared CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # infer conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, n, n)
            conv_out_dim = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU()
        )
        # Policy & Value heads
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.policy(x), self.value(x)

    def act(self, state):
        """Sample an action given state (expects shape (1,C,H,W))"""
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(self, states, actions):
        """Evaluate states and actions for PPO update"""
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


# =====================
# Rollout Buffer
# =====================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()  # reset buffer


# =====================
# GAE Advantage Estimation
# =====================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    next_value = 0

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]

    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


# =====================
# PPO Agent
# =====================
class PPOTrainer:
    def __init__(self, env, input_channels=3, action_dim=4, n=20,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, update_epochs=10, batch_size=64):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.env = env

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(input_channels, action_dim, n=n).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        """
        state: numpy array (C,H,W) or (H,W,C)
        """
        s = torch.tensor(state, dtype=torch.float32).to(DEVICE)

        # fix dimensions if needed: (H,W,C) -> (C,H,W)
        if s.dim() == 3 and s.shape[0] not in [1, 3]:
            s = s.permute(2, 0, 1)

        s = s.unsqueeze(0)  # add batch dim (1,C,H,W)

        action, log_prob, value = self.policy.act(s)
        self.buffer.states.append(s)
        self.buffer.actions.append(torch.tensor(action))
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value.item())
        return action

    def store_outcome(self, reward, done):
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def update(self):
        # Convert buffer to tensors
        states = torch.cat(self.buffer.states).to(DEVICE)
        actions = torch.stack(self.buffer.actions).to(DEVICE)
        old_log_probs = torch.stack(self.buffer.log_probs).detach().to(DEVICE)
        values = torch.tensor(self.buffer.values, dtype=torch.float32).to(DEVICE)

        # Compute GAE
        advantages, returns = compute_gae(
            self.buffer.rewards, self.buffer.values, self.buffer.dones,
            gamma=self.gamma, lam=self.lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages, returns = advantages.to(DEVICE), returns.to(DEVICE)

        # PPO Update
        for _ in range(self.update_epochs):
            for i in range(0, len(states), self.batch_size):
                idx = slice(i, i + self.batch_size)

                log_probs, values_pred, entropy = self.policy.evaluate(states[idx], actions[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])

                # Clipped surrogate objective
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = nn.MSELoss()(values_pred, returns[idx])

                # Total loss
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        total_reward = sum(self.buffer.rewards)

        self.buffer.clear()
        return total_reward

    def train(self, n_episodes=2000, max_steps=400):
        total_reward_out = []
        for ep in range(n_episodes):
            state = self.env.reset()
            done = False

            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store_outcome(reward, done)
                state = next_state
                if done: break

            total_reward = self.update()
            total_reward_out.append(total_reward)
            print(f"Episode {ep}, total reward = {total_reward}")
        return total_reward_out

class PPO_Agent:
    def __init__(self, model_path=None, input_channels=3, n_actions=4):
        self.model = ActorCritic(input_channels, n_actions).to(DEVICE)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()  # no dropout/bn in eval mode

    def act(self, state):
        """
        Pick the greedy action for evaluation.
        state: numpy array (H, W, C) or (C, H, W)
        returns: int (action index)
        """
        s = torch.from_numpy(state).float().to(DEVICE)

        # Fix dimensions: (H,W,C) -> (C,H,W)
        if s.dim() == 3 and s.shape[0] not in [1, 3]:
            s = s.permute(2, 0, 1)

        s = s.unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            logits, _ = self.model(s)
            action = torch.argmax(logits, dim=1).item()

        return action
