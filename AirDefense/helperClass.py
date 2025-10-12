import gymnasium as gym
import numpy as np

class MultiAgentWrapper(gym.Env):
    def __init__(self, env, agent_type="good"):
        super().__init__()
        self.env = env
        self.agent_type = agent_type

        # Pick agents of the selected type
        self.agents = [a for a in env.agents if agent_type in a]
        sample_obs = env.observation_space(self.agents[0])
        sample_act = env.action_space(self.agents[0])
        self.observation_space = sample_obs
        self.action_space = sample_act

    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        obs = np.stack([obs_dict[a] for a in self.agents])
        return obs, {}

    def step(self, actions):
        # Map back to dict for underlying env
        actions_dict = {a: actions[i] for i, a in enumerate(self.agents)}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(actions_dict)

        obs = np.stack([obs_dict[a] for a in self.agents])
        rewards = np.array([rew_dict[a] for a in self.agents])
        done = all(term_dict.values()) or all(trunc_dict.values())
        infos = {}
        return obs, rewards, done, infos
