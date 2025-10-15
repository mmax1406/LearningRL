import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import optuna
from collections import deque

GOOD_MODEL_PATH = "good_policy.pt"
TRAINED_ADVERSARY_PATH = "adversary_policy.pt"
TOTAL_TIMESTEPS = 200_000
BATCH_SIZE = 512
ROLLOUT_STEPS = 2048
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
UPDATE_EPOCHS = 4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
EVAL_FREQ = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def act(self, obs):
        mean = self.actor(obs)
        dist = Normal(mean, torch.ones_like(mean) * 0.1)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(-1, 1), log_prob

    def evaluate(self, obs, actions):
        mean = self.actor(obs)
        dist = Normal(mean, torch.ones_like(mean) * 0.1)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    advantages = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages), torch.tensor(returns)


def train_adversary_agent(trial=None):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) if trial else 3e-4
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.05, log=True) if trial else ENTROPY_COEF
    value_coef = trial.suggest_float("value_coef", 0.1, 1.0) if trial else VALUE_COEF

    from intercept_env import env
    environment = env(N_adversaries=3, M_good=5, width_ratio=3.0)

    obs_dict, _ = environment.reset()
    adv_agents = [a for a in environment.agents if "adversary" in a]
    good_agents = [a for a in environment.agents if "good" in a]

    obs_dim = len(obs_dict[adv_agents[0]])
    act_dim = environment.action_space(adv_agents[0]).shape[0]

    policy = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Load good agent if exists
    good_policy = None
    if os.path.exists(GOOD_MODEL_PATH):
        good_policy = ActorCritic(obs_dim, act_dim).to(DEVICE)
        good_policy.load_state_dict(torch.load(GOOD_MODEL_PATH))
        good_policy.eval()
        print("✅ Loaded good agent policy")
    else:
        print("⚠️ No good model found, using random actions")

    ep_rewards = deque(maxlen=10)

    for step in range(1, TOTAL_TIMESTEPS + 1):
        rollout = []
        obs_dict, _ = environment.reset()

        for _ in range(ROLLOUT_STEPS):
            actions = {}
            step_data = []
            for a in adv_agents:
                obs = torch.tensor(obs_dict[a], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    action, log_prob = policy.act(obs)
                actions[a] = action.cpu().numpy()
                step_data.append((obs, action, log_prob))
            for a in good_agents:
                if good_policy:
                    obs = torch.tensor(obs_dict[a], dtype=torch.float32, device=DEVICE)
                    with torch.no_grad():
                        action, _ = good_policy.act(obs)
                    actions[a] = action.cpu().numpy()
                else:
                    actions[a] = environment.action_space(a).sample()

            next_obs, rewards, terms, truncs, infos = environment.step(actions)
            done = all(terms.values()) or len(environment.agents) == 0
            reward = np.mean([rewards[a] for a in adv_agents])
            rollout.append((step_data, reward, done))
            obs_dict = next_obs if not done else environment.reset()[0]

        states, actions_t, log_probs_old, rewards, dones = [], [], [], [], []
        for (step_data, r, d) in rollout:
            for obs, act, logp in step_data:
                states.append(obs)
                actions_t.append(act)
                log_probs_old.append(logp)
                rewards.append(r)
                dones.append(float(d))

        states = torch.stack(states)
        actions_t = torch.stack(actions_t)
        log_probs_old = torch.stack(log_probs_old)
        with torch.no_grad():
            values = policy.critic(states).squeeze(-1)
        advantages, returns = compute_gae(rewards, values.cpu(), dones)

        for _ in range(UPDATE_EPOCHS):
            idx = np.random.permutation(len(states))
            for start in range(0, len(states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch = idx[start:end]
                log_probs, entropy, values = policy.evaluate(states[batch], actions_t[batch])
                ratio = (log_probs - log_probs_old[batch]).exp()
                adv = advantages[batch].to(DEVICE)
                ret = returns[batch].to(DEVICE)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values - ret).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if step % EVAL_FREQ == 0:
            avg_reward = np.mean(rewards)
            ep_rewards.append(avg_reward)
            print(f"Step {step}: Avg Adversary Reward = {avg_reward:.2f}")
            if trial and len(ep_rewards) >= 5:
                return np.mean(ep_rewards)

    torch.save(policy.state_dict(), TRAINED_ADVERSARY_PATH)
    print(f"✅ Saved trained adversary policy at {TRAINED_ADVERSARY_PATH}")
    return np.mean(ep_rewards) if len(ep_rewards) > 0 else 0.0


if __name__ == "__main__":
    use_optuna = True
    if use_optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(train_adversary_agent, n_trials=10)
        print("Best params:", study.best_params)
    else:
        train_adversary_agent()
