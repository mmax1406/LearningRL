import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, input_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # compute size after flatten
        dummy = torch.zeros(1, input_channels, 20, 20)
        n_flatten = self.conv(dummy).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(n_flatten, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.actor(x), self.critic(x)


# --- A2C Agent ---
class A2CTrainer:
    def __init__(self, env, gamma=0.99, lr=1e-4, entropy_beta=0.01):
        self.env = env
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.n_actions = 4
        self.model = ActorCritic(input_channels=3, n_actions=self.n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)  # [1,3,n,n]
        logits, _ = self.model(state_t)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def train(self, n_episodes=2000, max_steps=400):
        total_reward_out, bestReward = [], -10000
        for ep in range(n_episodes):
            state = self.env.reset()  # shape [3,n,n]
            total_reward, log_probs, values, rewards, entropies = 0, [], [], [], []

            for step in range(max_steps):
                action, log_prob, entropy = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # evaluate value of current state
                state_t = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)
                _, value = self.model(state_t)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)

                total_reward += reward
                state = next_state

                if done:
                    break

            # --- Compute Returns ---
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
            values = torch.cat(values).squeeze(-1)
            log_probs = torch.stack(log_probs)
            entropies = torch.stack(entropies)

            advantages = returns - values.detach()

            # --- Loss ---
            actor_loss = -(log_probs * advantages).mean() - self.entropy_beta * entropies.mean()
            critic_loss = nn.MSELoss()(values, returns)
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_reward_out.append(total_reward)
            print(f"Episode {ep+1}: Reward {total_reward:.2f}")

            if total_reward > bestReward and ep > 300:
                torch.save(self.model.state_dict(), f"weights.pth")
                bestReward = total_reward

        return total_reward_out

class A2C_Agent:
    def __init__(self, model_path=None, input_channels=3, n_actions=4):
        self.model = ActorCritic(input_channels, n_actions).to(DEVICE)
        self.model.load_state_dict(torch.load("weights.pth"))
        self.model.eval()  # evaluation mode (no dropout/bn)

    def act(self, state):
        """
        Pick the greedy action for evaluation.
        state: numpy array (H, W, C) or (C, H, W)
        returns: int (action index)
        """
        # convert numpy -> tensor
        s = torch.from_numpy(state).float().to(DEVICE)

        # fix dimensions: (H,W,C) -> (C,H,W)
        if s.dim() == 3 and s.shape[0] != 3:
            s = s.permute(2, 0, 1)

        s = s.unsqueeze(0)  # add batch dim

        with torch.no_grad():
            logits, _ = self.model(s)
            action = torch.argmax(logits, dim=1).item()

        return action
