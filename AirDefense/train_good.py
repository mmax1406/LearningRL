import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from intercept_env import PursuitEvasionEnv
import gymnasium as gym

ADVERSARY_MODEL_PATH = "adversary_policy.zip"
GOOD_MODEL_PATH = "good_policy.zip"


class SharedGoodAgentsEnv(gym.Env):
    """
    Wraps your multi-agent environment to make it compatible with Stable Baselines3 PPO.
    All 'good' agents share one policy network.
    Adversaries act randomly or using a preloaded model.
    """
    def __init__(self, base_env, good_model=None):
        super().__init__()
        self.base_env = base_env
        self.good_model = good_model
        self.agents = base_env.agents
        self.good_agents = [a for a in self.agents if "good" in a]
        self.adv_agents = [a for a in self.agents if "adversary" in a]

        obs_sample = self.base_env._get_obs()[self.agents[0]]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(base_env.action_space(self.agents[0]).n)

    def reset(self):
        obs = self.base_env.reset()
        self.current_obs = obs
        # Return observation of first good agent (shared policy)
        return obs[self.good_agents[0]]

    def step(self, action):
        # Step all good agents using same policy action
        actions = {}
        for agent in self.good_agents:
            actions[agent] = action

        # Adversaries: random or fixed model
        for agent in self.adv_agents:
            obs = self.current_obs[agent].reshape(1, -1)
            if self.good_model:
                act, _ = self.good_model.predict(obs, deterministic=True)
                actions[agent] = act[0]
            else:
                actions[agent] = self.base_env.action_space(agent).sample()

        # Step base env
        obs, rewards, dones, truncs, infos = self.base_env.step(actions)
        self.current_obs = obs

        # Aggregate rewards of all good agents
        total_reward = np.mean([rewards[a] for a in self.good_agents])
        done = all(dones.values())
        trunc = any(truncs.values())

        return obs[self.good_agents[0]], total_reward, done, trunc, {}

# ---------------------------------------------------------
# Helper: Create training environment
# ---------------------------------------------------------
def make_env(N_good=3, N_adv=2, width_ratio=5.0):
    """
    Returns a PursuitEvasionEnv instance.
    Adversaries use pretrained weights if available, otherwise random.
    """
    env = PursuitEvasionEnv(N_adversaries=N_adv, M_good=N_good, width_ratio=width_ratio)

    # Load adversary model if available
    if os.path.exists(ADVERSARY_MODEL_PATH):
        print("✅ Loading adversary model as opponents...")
        adv_model = PPO.load(ADVERSARY_MODEL_PATH)

        # Override env.step() to use adversary policy
        original_step = env.step

        def wrapped_step(actions):
            obs = env._get_obs()
            # Add adversary actions from model
            for agent in env.agents:
                if "adversary" in agent and agent not in actions:
                    obs_vec = obs[agent].reshape(1, -1)
                    act, _ = adv_model.predict(obs_vec, deterministic=True)
                    actions[agent] = act[0]
            return original_step(actions)

        env.step = wrapped_step
    else:
        print("⚠️ No adversary model found, using random actions instead.")

        original_step = env.step

        def wrapped_step(actions):
            obs = env._get_obs()
            for agent in env.agents:
                if "adversary" in agent and agent not in actions:
                    # Random adversary actions
                    low, high = env.action_spaces[agent].low, env.action_spaces[agent].high
                    actions[agent] = np.random.uniform(low, high)
            return original_step(actions)

        env.step = wrapped_step

    return env

# ---------------------------------------------------------
# Training logic
# ---------------------------------------------------------
def train_good(N_good=3, N_adv=2, total_timesteps=1_000_000):
    print(f"🚀 Training good agents ({N_good}) vs adversaries ({N_adv})")

    base_env = make_env(N_good, N_adv)
    adv_model = None
    if os.path.exists(ADVERSARY_MODEL_PATH):
        print("✅ Using pre-trained adversary model")
        adv_model = PPO.load(ADVERSARY_MODEL_PATH)

    env = SharedGoodAgentsEnv(base_env, good_model=adv_model)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models", exist_ok=True)
    model.save(GOOD_MODEL_PATH)
    print("✅ Saved good agent model.")

if __name__ == "__main__":
    train_good(N_good=3, N_adv=2)
