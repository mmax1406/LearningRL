import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
import supersuit as ss

# Main enviorment script

class PursuitEvasionEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "intercept_v2"}

    def __init__(self, N_adversaries=2, M_good=2, width_ratio=5.0, render_mode=None):
        self.N_adversaries = N_adversaries
        self.M_good = M_good
        self.agents = [f"adversary_{i}" for i in range(N_adversaries)] + [
            f"good_{i}" for i in range(M_good)
        ]
        self.possible_agents = self.agents.copy()
        self.pos = {}
        self.vel = {}
        self.active = {a: True for a in self.agents}
        self.agent_radius = 0.04
        self.width_ratio = width_ratio
        self.render_mode = render_mode
        self.viewer = None

        # World dimensions
        self.size_x = width_ratio * 2
        self.size_y = 2

        # --- Action spaces (acceleration control) ---
        self.action_spaces = {
            **{
                f"adversary_{i}": gym.spaces.Box(
                    low=np.array([-1.0, -1.0]),
                    high=np.array([1.0, 1.0]),
                    dtype=np.float32,
                )
                for i in range(N_adversaries)
            },
            **{
                f"good_{i}": gym.spaces.Box(
                    low=np.array([-1.0, -1.0]),
                    high=np.array([1.0, 1.0]),
                    dtype=np.float32,
                )
                for i in range(M_good)
            },
        }

        # --- Observation space ---
        # [own pos (2), own vel (2), nearest adversary (2), nearest good (2),
        # bounding box of adversaries (4), bounding box of good agents (4)] = 16 dims
        obs_dim = 16
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.observation_spaces = {a: obs_space for a in self.agents}

    # -----------------------------------------------------
    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self.agents = [f"adversary_{i}" for i in range(self.N_adversaries)] + [
            f"good_{i}" for i in range(self.M_good)
        ]
        self.active = {a: True for a in self.agents}

        # --- Initialize positions and velocities ---
        for a in self.agents:
            if "adversary" in a:
                self.pos[a] = np.array(
                    [self.width_ratio - 0.2, np.random.uniform(-0.8, 0.8)]
                )
                self.vel[a] = np.array([-0.02, 0.0])
            else:
                self.pos[a] = np.array(
                    [-self.width_ratio + 0.2, np.random.uniform(-0.8, 0.8)]
                )
                self.vel[a] = np.array([0.02, 0.0])

        return self._get_obs(), {a: {} for a in self.agents}

    # -----------------------------------------------------
    def step(self, actions):
        for a, act in actions.items():
            if not self.active.get(a, False):
                continue

            act = np.clip(act, self.action_spaces[a].low, self.action_spaces[a].high)
            self.vel[a] += 0.02 * act
            self.pos[a] += self.vel[a]

            # Clamp inside screen
            self.pos[a][0] = np.clip(
                self.pos[a][0],
                -self.width_ratio + self.agent_radius,
                self.width_ratio - self.agent_radius,
            )
            self.pos[a][1] = np.clip(
                self.pos[a][1], -1 + self.agent_radius, 1 - self.agent_radius
            )

        # --- Rewards ---
        rewards = {a: -0.01 for a in self.agents}  # time penalty
        terminations = {a: False for a in self.agents}
        good_reward, adv_reward = 0.0, 0.0

        # --- Collisions ---
        for g in [a for a in self.agents if "good" in a]:
            for adv in [a for a in self.agents if "adversary" in a]:
                if not (self.active[g] and self.active[adv]):
                    continue
                dist = np.linalg.norm(self.pos[g] - self.pos[adv])
                if dist < 2 * self.agent_radius:
                    self.active[g] = False
                    self.active[adv] = False
                    good_reward += 10.0
                    adv_reward -= 10.0

        # --- Adversaries reach left boundary ---
        for adv in [a for a in self.agents if "adversary" in a and self.active[a]]:
            if self.pos[adv][0] < -self.width_ratio + 0.1:
                self.active[adv] = False
                good_reward -= 10.0
                adv_reward += 10.0

        # --- Assign team rewards ---
        for a in self.agents:
            if "good" in a:
                rewards[a] += good_reward
            else:
                rewards[a] += adv_reward

        # --- Termination if all adversaries inactive ---
        if all(not self.active[a] for a in self.agents if "adversary" in a):
            for a in self.agents:
                terminations[a] = True
            self.agents = []

        return (
            self._get_obs(),
            rewards,
            terminations,
            {a: self.active[a] for a in self.agents},  # truncations
            {a: {} for a in self.agents},     # infos
        )

    # -----------------------------------------------------
    def _get_obs(self):
        obs = {}

        # Get team arrays for bounding boxes
        adv_positions = np.array(
            [self.pos[a] for a in self.agents if "adversary" in a and self.active[a]]
        )
        good_positions = np.array(
            [self.pos[a] for a in self.agents if "good" in a and self.active[a]]
        )

        if len(adv_positions) > 0:
            adv_box = np.array(
                [
                    np.min(adv_positions[:, 0]),
                    np.max(adv_positions[:, 0]),
                    np.min(adv_positions[:, 1]),
                    np.max(adv_positions[:, 1]),
                ]
            )
        else:
            adv_box = np.zeros(4)

        if len(good_positions) > 0:
            good_box = np.array(
                [
                    np.min(good_positions[:, 0]),
                    np.max(good_positions[:, 0]),
                    np.min(good_positions[:, 1]),
                    np.max(good_positions[:, 1]),
                ]
            )
        else:
            good_box = np.zeros(4)

        # --- Build observation per agent ---
        for a in self.agents:
            if not self.active.get(a, False):
                obs[a] = np.zeros(16, dtype=np.float32)
                continue

            own_pos = self.pos[a]
            own_vel = self.vel[a]

            # Find nearest adversary and nearest good agent
            if "good" in a:
                others_adv = [
                    self.pos[o] for o in self.agents if "adversary" in o and self.active[o]
                ]
                others_good = [
                    self.pos[o] for o in self.agents if "good" in o and self.active[o] and o != a
                ]
            else:
                others_adv = [
                    self.pos[o] for o in self.agents if "adversary" in o and self.active[o] and o != a
                ]
                others_good = [
                    self.pos[o] for o in self.agents if "good" in o and self.active[o]
                ]

            # Handle missing opponents
            nearest_adv = np.zeros(2)
            nearest_good = np.zeros(2)

            if len(others_adv) > 0:
                nearest_adv = others_adv[np.argmin(np.linalg.norm(np.array(others_adv) - own_pos, axis=1))] - own_pos
            if len(others_good) > 0:
                nearest_good = others_good[np.argmin(np.linalg.norm(np.array(others_good) - own_pos, axis=1))] - own_pos

            obs[a] = np.concatenate(
                [own_pos, own_vel, nearest_adv, nearest_good, adv_box, good_box]
            ).astype(np.float32)

        return obs

    # -----------------------------------------------------
    def render(self):
        import matplotlib.pyplot as plt

        plt.clf()
        for a in self.agents:
            if not self.active[a]:
                continue
            color = "red" if "adversary" in a else "blue"
            plt.scatter(self.pos[a][0], self.pos[a][1], c=color, s=100)
        plt.xlim(-self.width_ratio, self.width_ratio)
        plt.ylim(-1, 1)
        plt.title("Intercept Environment")
        plt.pause(0.01)


def env(**kwargs):
    return PursuitEvasionEnv(**kwargs)
