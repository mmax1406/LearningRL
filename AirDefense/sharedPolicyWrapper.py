import numpy as np
import gymnasium as gym

class SharedPolicyWrapper(gym.Env):
    def __init__(self, pettingzoo_env):
        super().__init__()
        self.env = pettingzoo_env
        self.env.reset()
        self.agents = self.env.agents

        # Assume all agents share the same obs/action spaces
        self.observation_space = self.env.observation_space(self.agents[0])
        self.action_space = self.env.action_space(self.agents[1])
        self.num_agents = len(self.agents)

    def reset(self, *, seed=None, options=None):
        obs_dict,_ = self.env.reset(seed=seed, options=options)
        obs = np.stack([obs_dict[a] for a in self.agents])
        return obs, {}

    def step(self, actions):
        # Clamp actions depending on agent type
        action_dict = {}
        for i, a in enumerate(self.agents):
            act = np.clip(actions[i], -1, 1)
            if "adversary" in a:
                act[0] = np.clip(act[0], -1, 0)
            action_dict[a] = act
        next_obs, rewards, terms, truncs, infos = self.env.step(action_dict)

        # Check if any agents are done or destroyed
        done = all(terms.values()) or len(self.env.agents) == 0
        obs = np.stack([next_obs[a] for a in self.agents if truncs[a]])
        reward = np.array([rewards[a] for a in self.agents])  # shared

        if done:
            print("")

        # I also need to return good & Bad to know who to train
        return obs, reward, done, infos