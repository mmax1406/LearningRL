import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv


class InterceptParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "intercept_v0"}

    def __init__(self, N_adversaries=2, M_good=2, render_mode="human"):
        super().__init__()
        self.N_adv = N_adversaries
        self.M_good = M_good
        self.render_mode = render_mode

        self.agents = [f"adversary_{i}" for i in range(N_adversaries)] + [f"good_{i}" for i in range(M_good)]
        self.agent_radius = 2
        self.pos = {}
        self.vel = {}
        self.active = {}
        self.timestep = 0

        # Action space: move in x/y [-1, 1]
        self.action_spaces = {a: spaces.Box(-1, 1, shape=(2,), dtype=np.float32) for a in self.agents}

        # Observation: own pos + velocity + other agents' pos
        obs_dim = 4 + 2 * (N_adversaries + M_good - 1)
        self.observation_spaces = {a: spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32) for a in self.agents}

        # Rendering
        self.size = 600
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.pos = {}
        self.vel = {}
        self.active = {}

        for i in range(self.N_adv):
            self.pos[f"adversary_{i}"] = np.array([0.8, 0.8]) + np.random.uniform(-0.05, 0.05, 2)
            self.vel[f"adversary_{i}"] = np.zeros(2)
            self.active[f"adversary_{i}"] = True

        for i in range(self.M_good):
            self.pos[f"good_{i}"] = np.array([-0.8, -0.8]) + np.random.uniform(-0.05, 0.05, 2)
            self.vel[f"good_{i}"] = np.zeros(2)
            self.active[f"good_{i}"] = True

        return self._get_obs()

    def step(self, actions):
        rewards = {a: 0 for a in self.agents}

        # update positions
        for a, act in actions.items():
            if not self.active.get(a, False):
                continue
            act = np.clip(act, -1, 1)
            self.pos[a] += 0.05 * act
            # clip position so agents stay on screen
            r = self.agent_radius / (self.size / 2)
            self.pos[a][0] = np.clip(self.pos[a][0], -1 + r, 1 - r)
            self.pos[a][1] = np.clip(self.pos[a][1], -1 + r, 1 - r)

        # check collisions
        to_remove = []
        for g in [x for x in self.agents if "good" in x and self.active[x]]:
            for adv in [x for x in self.agents if "adversary" in x and self.active[x]]:
                if np.linalg.norm(self.pos[g] - self.pos[adv]) < 0.05:
                    to_remove += [g, adv]
                    rewards[g] += 10
                    rewards[adv] -= 10

        for r in to_remove:
            self.active[r] = False

        # check adversaries reaching bottom-left
        for adv in [x for x in self.agents if "adversary" in x and self.active[x]]:
            if np.linalg.norm(self.pos[adv] - np.array([-0.8, -0.8])) < 0.05:
                rewards[adv] += 20
                self.active[adv] = False

        self.timestep += 1
        obs = self._get_obs()
        terminations = {a: not self.active.get(a, False) for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

    def _get_obs(self):
        obs = {}
        for a in self.agents:
            if not self.active.get(a, False):
                obs[a] = np.zeros_like(list(self.observation_spaces.values())[0].low)
                continue
            other_pos = [self.pos[o] - self.pos[a] for o in self.agents if o != a and self.active[o]]
            obs[a] = np.concatenate([self.pos[a], self.vel[a]] + other_pos)
        return obs

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.size, self.size))
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))
        for a in self.agents:
            if not self.active.get(a, False):
                continue
            color = (200, 50, 50) if "adversary" in a else (50, 200, 50)
            px = int((self.pos[a][0] + 1) * self.size / 2)
            py = int((1 - self.pos[a][1]) * self.size / 2)
            pygame.draw.circle(self.window, color, (px, py), self.agent_radius)
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None


def env(**kwargs):
    return InterceptParallelEnv(**kwargs)
